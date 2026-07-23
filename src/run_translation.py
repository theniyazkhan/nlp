"""
run_translation.py – Translate FLORES+ eng_Latn devtest with NLLB.
Designed to run on a Kaggle T4 GPU.

Usage:
    python src/run_translation.py \\
        --model facebook/nllb-200-distilled-600M \\
        --languages-file languages.txt \\
        --output-dir results/translations \\
        --batch-size 32

Arguments:
    --model          HuggingFace model id (default: facebook/nllb-200-distilled-600M)
    --languages-file Path to file with one NLLB language code per line
    --output-dir     Directory for output JSONL files
    --batch-size     Number of sentences per translation batch (default: 32)
    --load-8bit      Load model in 8-bit quantisation (requires bitsandbytes)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


# ─────────────────────────────── CLI ────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Translate FLORES+ eng_Latn devtest into target languages via NLLB."
    )
    p.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="HuggingFace model ID (default: facebook/nllb-200-distilled-600M)",
    )
    p.add_argument(
        "--languages-file",
        default="languages.txt",
        help="File with one NLLB language code per line (default: languages.txt)",
    )
    p.add_argument(
        "--output-dir",
        default="results/translations",
        help="Output directory (default: results/translations)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Translation batch size (default: 32)",
    )
    p.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit (requires bitsandbytes)",
    )
    return p.parse_args()


# ─────────────────────────────── Helpers ────────────────────────────────────
def load_sentences(hf_token: str) -> list[dict]:
    """Load eng_Latn devtest from FLORES+, return list of {sentence_id, text}."""
    from datasets import load_dataset
    from huggingface_hub import login

    login(token=hf_token, add_to_git_credential=False)
    print("Loading FLORES+ eng_Latn devtest …")
    ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    records = [{"sentence_id": i, "text": row["text"]} for i, row in enumerate(ds)]
    print(f"  Loaded {len(records)} sentences.")
    return records


def load_languages(path: str) -> list[str]:
    """Read language codes from file, one per line, skip blanks."""
    langs = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return langs


def count_lines(path: Path) -> int:
    """Count non-empty lines in a file (used for resumability check)."""
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def batched(lst: list, size: int):
    """Yield successive slices of `lst` of length `size`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def load_model_and_tokenizer(model_id: str, load_8bit: bool):
    """Load NLLB model + tokenizer. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading model '{model_id}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if load_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    model.eval()
    print(f"  Model loaded (8-bit={load_8bit}).")
    return model, tokenizer


def translate_language(
    lang_code: str,
    source_records: list[dict],
    model,
    tokenizer,
    batch_size: int,
    out_path: Path,
) -> None:
    """Translate all source sentences into `lang_code` and write to `out_path`."""
    import torch

    device = next(model.parameters()).device
    target_lang = lang_code  # NLLB uses the flores+ codes directly

    with open(out_path, "w", encoding="utf-8") as fout:
        for batch in tqdm(
            batched(source_records, batch_size),
            total=(len(source_records) + batch_size - 1) // batch_size,
            desc=lang_code,
            leave=False,
        ):
            sources = [r["text"] for r in batch]

            inputs = tokenizer(
                sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                    max_length=512,
                    num_beams=4,
                )

            translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for rec, hyp in zip(batch, translations):
                out_record = {
                    "sentence_id": rec["sentence_id"],
                    "source": rec["text"],
                    "hypothesis": hyp,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")


# ─────────────────────────────── Main ───────────────────────────────────────
def main() -> None:
    args = get_args()

    # ── HF authentication ──────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")

    # ── Load sentences ─────────────────────────────────────────────────────
    source_records = load_sentences(hf_token)
    n_expected = len(source_records)

    # ── Language list ──────────────────────────────────────────────────────
    languages = load_languages(args.languages_file)
    print(f"Languages to translate: {len(languages)}")

    # ── Output directory ───────────────────────────────────────────────────
    model_short = args.model.replace("/", "_")
    out_dir = Path(args.output_dir) / model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model, args.load_8bit)

    # ── Translate ──────────────────────────────────────────────────────────
    for lang in languages:
        out_path = out_dir / f"{lang}.jsonl"

        # Resumability: skip if file already has the expected number of lines
        if count_lines(out_path) == n_expected:
            print(f"[SKIP] {lang}: {out_path} already has {n_expected} lines.")
            continue

        print(f"\n[TRANSLATE] {lang} → {out_path}")
        t0 = time.perf_counter()
        translate_language(lang, source_records, model, tokenizer, args.batch_size, out_path)
        elapsed = time.perf_counter() - t0
        n_written = count_lines(out_path)
        print(f"  Done: {n_written} lines written in {elapsed:.1f}s")

    print("\nAll languages complete.")


if __name__ == "__main__":
    main()
