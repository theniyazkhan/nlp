"""
Translate FLORES+ eng_Latn devtest into target languages using NLLB models.
Saves jsonl translations per target language, supporting batch processing and resumable execution.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Translate FLORES+ eng_Latn devtest into target languages via NLLB."
    )
    p.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="HuggingFace model ID",
    )
    p.add_argument(
        "--languages-file",
        default="languages.txt",
        help="File with target language codes",
    )
    p.add_argument(
        "--output-dir",
        default="results/translations",
        help="Output directory",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Translation batch size",
    )
    p.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    return p.parse_args()


def load_sentences(hf_token: str) -> list[dict]:
    from datasets import load_dataset
    from huggingface_hub import login

    login(token=hf_token, add_to_git_credential=False)
    ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    return [{"sentence_id": i, "text": row["text"]} for i, row in enumerate(ds)]


def load_languages(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def batched(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def load_model_and_tokenizer(model_id: str, load_8bit: bool):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

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
    return model, tokenizer


def translate_language(
    lang_code: str,
    source_records: list[dict],
    model,
    tokenizer,
    batch_size: int,
    out_path: Path,
) -> None:
    import torch

    device = next(model.parameters()).device
    target_lang = lang_code

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


def main() -> None:
    args = get_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")

    source_records = load_sentences(hf_token)
    n_expected = len(source_records)

    languages = load_languages(args.languages_file)
    model_short = args.model.replace("/", "_")
    out_dir = Path(args.output_dir) / model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, args.load_8bit)

    for lang in languages:
        out_path = out_dir / f"{lang}.jsonl"

        if count_lines(out_path) == n_expected:
            print(f"[SKIP] {lang}: {out_path} already has {n_expected} lines.")
            continue

        print(f"\n[TRANSLATE] {lang} -> {out_path}")
        t0 = time.perf_counter()
        translate_language(lang, source_records, model, tokenizer, args.batch_size, out_path)
        elapsed = time.perf_counter() - t0
        n_written = count_lines(out_path)
        print(f"  Done: {n_written} lines written in {elapsed:.1f}s")

    print("\nAll languages complete.")


if __name__ == "__main__":
    main()
