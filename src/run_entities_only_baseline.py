"""
Evaluates FLORES+ translation benchmark against an entities-only copy baseline.
Extracts entity strings from English sentences and measures BLEU/chrF++ overlap against target language references.
Supports filtering entity types into names, numeric, or all entities.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

from resource_tiers import RESOURCE_TIERS

NAME_TYPES = {"PERSON", "GPE", "ORG", "LOC", "NORP", "FAC"}
NUMERIC_TYPES = {"DATE", "CARDINAL", "ORDINAL", "TIME", "MONEY", "PERCENT", "QUANTITY"}

LANG_CONFIG_MAPPING = {
    "zho_Hans": "cmn_Hans",
    "zho_Hant": "cmn_Hant",
}


def get_args():
    parser = argparse.ArgumentParser(description="Entities-Only Baseline Evaluation")
    parser.add_argument("--languages-file", type=str, default="languages.txt", help="Path to languages file")
    parser.add_argument("--entities-file", type=str, default="results/english_entities_trf.jsonl", help="Path to entities file")
    parser.add_argument(
        "--entity-types",
        type=str,
        choices=["names", "numeric", "all", "all-modes"],
        default="names",
        help="Entity types to extract: names, numeric, all, or all-modes (default: names)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    return parser.parse_args()


def load_languages(filepath="languages.txt"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Language file not found: {filepath}")

    languages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                languages.append(line)
    return languages


def load_entities(filepath: str, mode: str) -> list[list[str]]:
    path = Path(filepath)
    if not path.exists() and filepath == "results/english_entities_trf.jsonl":
        fallback_path = Path("english_entities.json")
        if fallback_path.exists():
            print(f"[Info] {path} not found. Falling back to {fallback_path}")
            path = fallback_path

    if not path.exists():
        raise FileNotFoundError(f"Entities file not found at {filepath} or fallback location.")

    entities_per_sentence = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                ents = record.get("entities", [])
                if mode == "names":
                    selected = [e["text"] for e in ents if e.get("label") in NAME_TYPES]
                elif mode == "numeric":
                    selected = [e["text"] for e in ents if e.get("label") in NUMERIC_TYPES]
                else:
                    selected = [e["text"] for e in ents]
                entities_per_sentence.append(selected)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for ents in data:
                entities_per_sentence.append(ents if isinstance(ents, list) else [])

    return entities_per_sentence


def get_hf_token():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        for env_path in [".env", os.path.expanduser("~/.env")]:
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                            val = line.split("=", 1)[1].strip("\"'")
                            if val:
                                return val
    if not token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")
    return token


def run_evaluation(entities_file: str, mode: str, languages_file: str, output_path: str):
    hf_token = get_hf_token()
    entities_per_sentence = load_entities(entities_file, mode)
    pseudo_hypotheses = [" ".join(ents) if ents else "" for ents in entities_per_sentence]
    target_languages = load_languages(languages_file)

    results = []

    for lang in tqdm(target_languages, desc=f"Evaluating entities-only ({mode})"):
        script = lang.split("_")[-1] if "_" in lang else "Unknown"
        resource_tier = RESOURCE_TIERS.get(lang, "unknown")
        config_lang = LANG_CONFIG_MAPPING.get(lang, lang)

        try:
            tgt_dataset = load_dataset(
                "openlanguagedata/flores_plus",
                config_lang,
                split="devtest",
                token=hf_token
            )
            sample_item = tgt_dataset[0]
            text_col = "text" if "text" in sample_item else "sentence"
            target_references = [row[text_col] for row in tgt_dataset]

            bleu_res = sacrebleu.corpus_bleu(pseudo_hypotheses, [target_references])
            chrf_res = sacrebleu.corpus_chrf(pseudo_hypotheses, [target_references], word_order=2)

            results.append({
                "language": lang,
                "script": script,
                "resource_tier": resource_tier,
                "entities_only_bleu": bleu_res.score,
                "entities_only_chrf": chrf_res.score
            })

        except Exception as e:
            print(f"\n[Warning] Failed to process language '{lang}': {e}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved results ({mode}) for {len(df)} languages to '{output_path}'.")
    return df


def compare_modes(mode_dfs: dict):
    print("\n" + "=" * 65)
    print("COMPARISON OF CHRF++ AVERAGES ACROSS ENTITY TYPE MODES")
    print("=" * 65)
    print(f"{'Entity Mode':<15} {'Languages':<12} {'Mean chrF++':<15} {'Mean BLEU':<15}")
    print("-" * 65)
    for mode, df in mode_dfs.items():
        mean_chrf = df["entities_only_chrf"].mean() if not df.empty else 0.0
        mean_bleu = df["entities_only_bleu"].mean() if not df.empty else 0.0
        print(f"{mode:<15} {len(df):<12} {mean_chrf:<15.4f} {mean_bleu:<15.4f}")
    print("=" * 65)


def main():
    args = get_args()
    if args.entity_types == "all-modes":
        mode_dfs = {}
        for m in ["names", "numeric", "all"]:
            out_file = f"results/entities_only_baseline_{m}.csv"
            mode_dfs[m] = run_evaluation(args.entities_file, m, args.languages_file, out_file)
        compare_modes(mode_dfs)
    else:
        output_file = args.output or f"results/entities_only_baseline_{args.entity_types}.csv"
        df = run_evaluation(args.entities_file, args.entity_types, args.languages_file, output_file)
        # Also create default results/entities_only_baseline.csv if mode is names for backwards compatibility
        if args.entity_types == "names" and not args.output:
            df.to_csv("results/entities_only_baseline.csv", index=False)


if __name__ == "__main__":
    main()
