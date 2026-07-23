# Results Directory

This directory contains output datasets and evaluation metrics generated during the FLORES+ named entity copying study.

## Output Files

| File Name | Description | Generating Script |
|---|---|---|
| `english_entities_trf.jsonl` | Named entities extracted from FLORES+ `eng_Latn` devtest using spaCy `en_core_web_trf`. | `src/extract_entities.py` |
| `copy_baseline_all_languages.csv` | Evaluation of verbatim English source copy baseline across target languages (BLEU & chrF++). | `src/run_copy_baseline.py` |
| `entities_only_baseline_names.csv` | Evaluation of proper-name pseudo-hypotheses baseline (default mode: PERSON, GPE, ORG, LOC, NORP, FAC). | `src/run_entities_only_baseline.py --entity-types names` |
| `entities_only_baseline_numeric.csv` | Evaluation of numeric pseudo-hypotheses baseline (DATE, CARDINAL, ORDINAL, TIME, MONEY, PERCENT, QUANTITY). | `src/run_entities_only_baseline.py --entity-types numeric` |
| `entities_only_baseline_all.csv` | Evaluation of pseudo-hypotheses baseline using all extracted entity types. | `src/run_entities_only_baseline.py --entity-types all` |
| `entities_only_baseline.csv` | Alias / default output copy of the proper-name baseline results. | `src/run_entities_only_baseline.py` |
| `baseline_comparison.csv` | Merged comparison of full-copy vs entities-only baselines and entity share of baseline score. | `src/compare_baselines.py` |
