"""
Evaluate FLORES+ English copy-baseline performance across target languages.
Calculates BLEU and chrF++ when English source sentences are directly copied as translation hypotheses.
"""

import argparse
import os
import sys
import pandas as pd
import sacrebleu
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="FLORES+ English Copy-Baseline Evaluation")
    parser.add_argument("--languages-file", type=str, default="languages.txt", help="Path to languages file")
    parser.add_argument("--output", type=str, default="copy_baseline_all_languages.csv", help="Output CSV path")
    return parser.parse_args()


def get_hf_token():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")
    return token


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


def main():
    args = get_args()
    hf_token = get_hf_token()
    login(token=hf_token, add_to_git_credential=False)

    eng_dataset = load_dataset(
        "openlanguagedata/flores_plus",
        "eng_Latn",
        split="devtest",
        token=hf_token
    )

    text_col = "text" if "text" in eng_dataset[0] else "sentence"
    english_hypotheses = [row[text_col] for row in eng_dataset]

    target_languages = load_languages(args.languages_file)
    results = []

    LANG_CONFIG_MAPPING = {
        "zho_Hans": "cmn_Hans",
        "zho_Hant": "cmn_Hant",
    }

    for lang in tqdm(target_languages, desc="Evaluating languages"):
        is_latin = lang.endswith("_Latn") or (len(lang.split("_")) > 1 and lang.split("_")[1] == "Latn")
        config_lang = LANG_CONFIG_MAPPING.get(lang, lang)

        try:
            tgt_dataset = load_dataset(
                "openlanguagedata/flores_plus",
                config_lang,
                split="devtest",
                token=hf_token
            )
            target_references = [row[text_col] for row in tgt_dataset]

            bleu_res = sacrebleu.corpus_bleu(english_hypotheses, [target_references])
            chrf_res = sacrebleu.corpus_chrf(english_hypotheses, [target_references], word_order=2)

            results.append({
                "language": lang,
                "is_latin": is_latin,
                "copy_bleu": bleu_res.score,
                "copy_chrf": chrf_res.score
            })

        except Exception as e:
            print(f"\n[Warning] Failed to process language {lang}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved evaluation results for {len(df)} languages to '{args.output}'.")

    latin_df = df[df["is_latin"] == True]
    non_latin_df = df[df["is_latin"] == False]

    avg_latin_bleu = latin_df["copy_bleu"].mean() if not latin_df.empty else 0.0
    avg_non_latin_bleu = non_latin_df["copy_bleu"].mean() if not non_latin_df.empty else 0.0
    avg_latin_chrf = latin_df["copy_chrf"].mean() if not latin_df.empty else 0.0
    avg_non_latin_chrf = non_latin_df["copy_chrf"].mean() if not non_latin_df.empty else 0.0

    print("\n" + "="*50)
    print("SUMMARY STATISTICS: COPY BASELINE EVALUATION")
    print("="*50)
    print(f"Total Languages Evaluated: {len(df)}")
    print(f"  - Latin Languages count:     {len(latin_df)}")
    print(f"  - Non-Latin Languages count: {len(non_latin_df)}")
    print("-" * 50)
    print(f"Average Copy BLEU (Latin):     {avg_latin_bleu:.4f}")
    print(f"Average Copy BLEU (Non-Latin): {avg_non_latin_bleu:.4f}")
    print(f"Average Copy chrF++ (Latin):     {avg_latin_chrf:.4f}")
    print(f"Average Copy chrF++ (Non-Latin): {avg_non_latin_chrf:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
