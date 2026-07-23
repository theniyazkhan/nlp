"""
Extract named entities from FLORES+ eng_Latn devtest using spaCy en_core_web_trf.
Saves entity records to JSONL format and optionally compares statistics against a previous extraction.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import spacy
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser(description="Extract NEs from FLORES+ using spaCy trf model")
    p.add_argument("--output", default="results/english_entities_trf.jsonl",
                   help="Output JSONL path (default: results/english_entities_trf.jsonl)")
    p.add_argument("--old-entities", default="english_entities.json",
                   help="Path to old english_entities.json for comparison")
    p.add_argument("--skip-if-exists", action="store_true",
                   help="Skip extraction if output file already exists")
    return p.parse_args()


def load_hf_and_dataset():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")
    login(token=token, add_to_git_credential=False)

    ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    sentences = [row["text"] for row in ds]
    return sentences


def print_comparison(new_records, old_entities_path):
    old_path = Path(old_entities_path)
    if not old_path.exists():
        return

    with open(old_path, encoding="utf-8") as f:
        old = json.load(f)

    old_counts = [len(x) for x in old]
    new_counts = [len(r["entities"]) for r in new_records]

    old_total = sum(old_counts)
    new_total = sum(new_counts)
    old_empty = sum(1 for c in old_counts if c == 0)
    new_empty = sum(1 for c in new_counts if c == 0)
    n = len(old_counts)

    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON: small vs trf model")
    print("=" * 60)
    print(f"{'Metric':<35} {'small':>10} {'trf':>10}")
    print("-" * 60)
    print(f"{'Total entities':<35} {old_total:>10,} {new_total:>10,}")
    print(f"{'Mean per sentence':<35} {old_total/n:>10.2f} {new_total/n:>10.2f}")
    print(f"{'% sentences with 0 entities':<35} {100*old_empty/n:>10.1f}% {100*new_empty/n:>10.1f}%")
    print("=" * 60)


def main():
    args = get_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_if_exists and out_path.exists():
        records = []
        label_counter = Counter()
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records.append(rec)
                    for e in rec.get("entities", []):
                        label_counter[e["label"]] += 1
    else:
        sentences = load_hf_and_dataset()
        model_name = "en_core_web_trf"
        try:
            nlp = spacy.load(model_name)
        except OSError:
            sys.exit(
                f"ERROR: spaCy model '{model_name}' not installed.\n"
                f"Run:  python -m spacy download {model_name}"
            )

        records = []
        label_counter = Counter()

        for sent_id, sentence in enumerate(tqdm(sentences, desc="Sentences")):
            doc = nlp(sentence)
            ents = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
                for ent in doc.ents
            ]
            for e in ents:
                label_counter[e["label"]] += 1
            records.append({"sentence_id": sent_id, "text": sentence, "entities": ents})

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(records)} records -> {out_path}")

    total = sum(len(r["entities"]) for r in records)
    n = len(records)
    empty = sum(1 for r in records if len(r["entities"]) == 0)

    print("\n" + "=" * 60)
    print("SUMMARY — en_core_web_trf")
    print("=" * 60)
    print(f"  Total entities          : {total:,}")
    print(f"  Mean per sentence       : {total/n:.2f}")
    print(f"  % sentences with 0 NEs : {100*empty/n:.1f}%  ({empty}/{n})")
    print("\n  Counts by label:")
    for label, count in sorted(label_counter.items(), key=lambda x: -x[1]):
        print(f"    {label:<15} {count:>6,}")
    print("=" * 60)

    print_comparison(records, args.old_entities)


if __name__ == "__main__":
    main()
