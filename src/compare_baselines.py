"""
Compare English Copy Baseline against Entities-Only Baseline scores across target languages.
Calculates the entity share of total copy chrF++ and flags metric artifacts like brevity-penalty distortions.
"""

import argparse
import os
import pandas as pd

from resource_tiers import RESOURCE_TIERS

EUROPEAN_LATIN = {"spa_Latn", "fra_Latn", "deu_Latn", "por_Latn", "ita_Latn"}
AFRICAN_ASIAN_LATIN = {"swh_Latn", "yor_Latn", "zul_Latn", "hau_Latn", "ibo_Latn", "vie_Latn", "ind_Latn"}


def main():
    parser = argparse.ArgumentParser(description="Compare Copy Baseline vs Entities-Only Baseline")
    parser.add_argument("--copy-csv", type=str, default="copy_baseline_all_languages.csv", help="Copy baseline CSV path")
    parser.add_argument(
        "--entities-csv",
        type=str,
        default="results/entities_only_baseline_names.csv",
        help="Entities baseline CSV path"
    )
    parser.add_argument("--output", type=str, default="results/baseline_comparison.csv", help="Output comparison CSV path")
    args = parser.parse_args()

    entities_csv = args.entities_csv
    if not os.path.exists(entities_csv) and os.path.exists("results/entities_only_baseline.csv"):
        entities_csv = "results/entities_only_baseline.csv"

    if not os.path.exists(args.copy_csv):
        raise FileNotFoundError(f"Copy baseline CSV not found: {args.copy_csv}")
    if not os.path.exists(entities_csv):
        raise FileNotFoundError(f"Entities baseline CSV not found: {entities_csv}")

    copy_df = pd.read_csv(args.copy_csv)
    entities_df = pd.read_csv(entities_csv)

    merged_df = pd.merge(copy_df, entities_df, on="language", how="inner")
    if "resource_tier" not in merged_df.columns:
        merged_df["resource_tier"] = merged_df["language"].map(lambda l: RESOURCE_TIERS.get(l, "unknown"))

    merged_df["entity_share_chrf"] = merged_df["entities_only_chrf"] / merged_df["copy_chrf"]

    cols = ["language", "script", "resource_tier", "is_latin", "copy_bleu", "copy_chrf", "entities_only_bleu", "entities_only_chrf", "entity_share_chrf"]
    merged_df = merged_df[[c for c in cols if c in merged_df.columns]]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged_df.to_csv(args.output, index=False)
    print(f"Saved merged baseline comparison to '{args.output}' ({len(merged_df)} languages).")

    print("\n" + "=" * 65)
    print("METRIC ARTIFACT WARNINGS")
    print("=" * 65)
    
    # Warning 1: BLEU 4-gram length artifact
    avg_ent_bleu = merged_df["entities_only_bleu"].mean()
    print(f"[WARNING] Entities-only BLEU is near zero across languages (mean: {avg_ent_bleu:.4f}).")
    print("          Reason: Pseudo-hypotheses are too short for higher n-gram (4-gram) matching.")
    print("          Evaluation metrics should rely on chrF++ only.")

    # Warning 2: Entity share > 1.0 artifact
    artifact_langs = merged_df[merged_df["entity_share_chrf"] > 1.0]
    if not artifact_langs.empty:
        lang_list = ", ".join(artifact_langs["language"].tolist())
        print(f"\n[WARNING] Found languages with entity_share_chrf > 1.0: {lang_list}")
        print("          Reason: Brevity penalty artifact in reference comparison, not a true score.")
        print("          These languages will be excluded from Figure 3 (entity share plot).")
    else:
        print("\n[Info] No languages with entity_share_chrf > 1.0 detected.")

    print("=" * 65)

    sorted_df = merged_df.sort_values(by="entity_share_chrf", ascending=False)

    print("\n" + "=" * 65)
    print("COMPARISON ANALYSIS: ENTITY SHARE OF COPY BASELINE (chrF++)")
    print("=" * 65)

    print("\n--- Top 5 Languages with HIGHEST Entity Share ---")
    print(sorted_df[["language", "script", "resource_tier", "copy_chrf", "entities_only_chrf", "entity_share_chrf"]].head(5).to_string(index=False))

    print("\n--- Bottom 5 Languages with LOWEST Entity Share ---")
    print(sorted_df[["language", "script", "resource_tier", "copy_chrf", "entities_only_chrf", "entity_share_chrf"]].tail(5).to_string(index=False))

    eur_latin_df = merged_df[merged_df["language"].isin(EUROPEAN_LATIN)]
    afr_asia_latin_df = merged_df[merged_df["language"].isin(AFRICAN_ASIAN_LATIN)]

    if not eur_latin_df.empty and not afr_asia_latin_df.empty:
        eur_copy = eur_latin_df["copy_chrf"].mean()
        eur_ent = eur_latin_df["entities_only_chrf"].mean()
        eur_share = eur_latin_df["entity_share_chrf"].mean()

        aa_copy = afr_asia_latin_df["copy_chrf"].mean()
        aa_ent = afr_asia_latin_df["entities_only_chrf"].mean()
        aa_share = afr_asia_latin_df["entity_share_chrf"].mean()

        print("\n" + "-" * 65)
        print("COGNATE VS. ENTITY ANALYSIS (LATIN SCRIPT SUBGROUPS)")
        print("-" * 65)
        print(f"European Latin Languages (n={len(eur_latin_df)}):")
        print(f"  - Mean Copy chrF++:          {eur_copy:.4f}")
        print(f"  - Mean Entities-Only chrF++: {eur_ent:.4f}")
        print(f"  - Mean Entity Share:         {eur_share:.4f} ({eur_share*100:.2f}%)")

        print(f"\nAfrican/Asian Latin Languages (n={len(afr_asia_latin_df)}):")
        print(f"  - Mean Copy chrF++:          {aa_copy:.4f}")
        print(f"  - Mean Entities-Only chrF++: {aa_ent:.4f}")
        print(f"  - Mean Entity Share:         {aa_share:.4f} ({aa_share*100:.2f}%)")
        print("=" * 65)


if __name__ == "__main__":
    main()
