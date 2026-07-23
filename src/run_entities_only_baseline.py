import os
import sys
import json
import argparse
import pandas as pd
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

# NOTE: Hardcoded resource tier mapping - NEEDS HUMAN REVIEW / VALIDATION
RESOURCE_TIERS = {
    # High Resource
    "spa_Latn": "high",
    "fra_Latn": "high",
    "deu_Latn": "high",
    "por_Latn": "high",
    "ita_Latn": "high",
    "zho_Hans": "high",
    "jpn_Jpan": "high",
    "rus_Cyrl": "high",
    "arb_Arab": "high",
    "kor_Hang": "high",
    # Mid Resource
    "ben_Beng": "mid",
    "hin_Deva": "mid",
    "urd_Arab": "mid",
    "tam_Taml": "mid",
    "tel_Telu": "mid",
    "mar_Deva": "mid",
    "npi_Deva": "mid",
    "vie_Latn": "mid",
    "ind_Latn": "mid",
    "tha_Thai": "mid",
    "khm_Khmr": "mid",
    # Low Resource
    "swh_Latn": "low",
    "yor_Latn": "low",
    "zul_Latn": "low",
    "hau_Latn": "low",
    "amh_Ethi": "low",
    "ibo_Latn": "low",
    "sin_Sinh": "low",
    "lao_Laoo": "low",
    "mya_Mymr": "low",
    "kat_Geor": "low",
    "khk_Cyrl": "low",
}

LANG_CONFIG_MAPPING = {
    "zho_Hans": "cmn_Hans",
    "zho_Hant": "cmn_Hant",
}


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
    parser = argparse.ArgumentParser(description="Entities-Only Baseline Evaluation")
    parser.add_argument("--languages-file", type=str, default="languages.txt", help="Path to languages file")
    parser.add_argument("--entities-file", type=str, default="english_entities.json", help="Path to English entities JSON file")
    parser.add_argument("--output", type=str, default="results/entities_only_baseline.csv", help="Output CSV path")
    args = parser.parse_args()

    # Read HF_TOKEN strictly from environment variable
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        # Check local .env files if environment variable is not directly exported
        for env_path in [".env", os.path.expanduser("~/.env")]:
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                            val = line.split("=", 1)[1].strip("\"'")
                            if val:
                                hf_token = val
                                break

    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set HF_TOKEN in environment or .env file.")

    # Load English entities JSON
    if not os.path.exists(args.entities_file):
        raise FileNotFoundError(f"Entities file not found: {args.entities_file}")

    with open(args.entities_file, "r", encoding="utf-8") as f:
        entities_per_sentence = json.load(f)

    # Build pseudo-translations containing ONLY English entity strings
    pseudo_hypotheses = [" ".join(ents) if ents else "" for ents in entities_per_sentence]
    print(f"Loaded {len(entities_per_sentence)} entity lists. Example pseudo-translation: '{pseudo_hypotheses[1]}'")

    target_languages = load_languages(args.languages_file)
    print(f"Loaded {len(target_languages)} target languages from '{args.languages_file}'.")

    results = []

    for lang in tqdm(target_languages, desc="Evaluating entities-only baseline"):
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

            # Compute BLEU and chrF++ (word_order=2)
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
            print(f"\n[Warning/Failure] Failed to process language '{lang}': {e}")

    # Ensure results/ directory exists and save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved entities-only baseline results for {len(df)} languages to '{args.output}'.")

    # Print summary statistics grouped by script and by resource_tier
    print("\n" + "="*60)
    print("SUMMARY STATISTICS: ENTITIES-ONLY BASELINE")
    print("="*60)

    print("\n--- Averages Grouped by Script ---")
    script_stats = df.groupby("script")[["entities_only_bleu", "entities_only_chrf"]].agg(["count", "mean"])
    print(script_stats.to_string())

    print("\n--- Averages Grouped by Resource Tier ---")
    tier_stats = df.groupby("resource_tier")[["entities_only_bleu", "entities_only_chrf"]].agg(["count", "mean"])
    print(tier_stats.to_string())

    print("="*60)


if __name__ == "__main__":
    main()
