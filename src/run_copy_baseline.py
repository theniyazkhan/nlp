import os
import sys
import argparse
import getpass

import pandas as pd
import sacrebleu
from datasets import load_dataset
from huggingface_hub import login, get_token
from tqdm import tqdm


def get_hf_token(cli_token=None):
    """
    Prompts for or reads the Hugging Face Token.
    Checks CLI argument, HF_TOKEN/HUGGING_FACE_HUB_TOKEN env vars,
    huggingface_hub cached token, .env files, and falls back to interactive prompt.
    """
    if cli_token:
        return cli_token

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    cached = get_token()
    if cached:
        return cached

    # Check local .env files
    for env_path in [".env", os.path.expanduser("~/.env")]:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                        val = line.split("=", 1)[1].strip("\"'")
                        if val:
                            return val

    # Interactive prompt fallback
    print("Hugging Face Token is required to access 'openlanguagedata/flores_plus'.")
    if sys.stdin.isatty():
        token = getpass.getpass("Enter your Hugging Face Token: ").strip()
    else:
        token = input("Enter your Hugging Face Token: ").strip()
    
    return token


def load_languages(filepath="languages.txt"):
    """Reads target languages from languages.txt."""
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
    parser = argparse.ArgumentParser(description="FLORES+ English Copy-Baseline Evaluation")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token")
    parser.add_argument("--languages-file", type=str, default="languages.txt", help="Path to languages file")
    parser.add_argument("--output", type=str, default="copy_baseline_all_languages.csv", help="Output CSV path")
    args = parser.parse_args()

    # 1. Obtain & authenticate with Hugging Face Token
    hf_token = get_hf_token(args.token)
    if not hf_token:
        raise ValueError("No Hugging Face Token provided.")

    login(token=hf_token, add_to_git_credential=False)

    # 2. Load FLORES+ English dataset (eng_Latn, split devtest)
    print("Loading FLORES+ English source dataset (eng_Latn, devtest)...")
    eng_dataset = load_dataset(
        "openlanguagedata/flores_plus",
        "eng_Latn",
        split="devtest",
        token=hf_token
    )

    # Extract text column
    sample_item = eng_dataset[0]
    text_col = "text" if "text" in sample_item else "sentence"
    english_hypotheses = [row[text_col] for row in eng_dataset]
    print(f"Loaded {len(english_hypotheses)} English sentences.")

    # 3. Read target languages & evaluate copy-baseline
    target_languages = load_languages(args.languages_file)
    print(f"Loaded {len(target_languages)} target languages from '{args.languages_file}'.")

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

            # Compute sacrebleu corpus BLEU and chrF++ (word_order=2)
            bleu_res = sacrebleu.corpus_bleu(english_hypotheses, [target_references])
            chrf_res = sacrebleu.corpus_chrf(english_hypotheses, [target_references], word_order=2)

            copy_bleu = bleu_res.score
            copy_chrf = chrf_res.score

            results.append({
                "language": lang,
                "is_latin": is_latin,
                "copy_bleu": copy_bleu,
                "copy_chrf": copy_chrf
            })

        except Exception as e:
            print(f"\n[Warning] Failed to process language {lang}: {e}")

    # 4. Save all outputs to copy_baseline_all_languages.csv
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved evaluation results for {len(df)} languages to '{args.output}'.")

    # 5. Print summary statistics comparing Latin vs. Non-Latin copy BLEU averages
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
