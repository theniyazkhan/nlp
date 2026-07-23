# FLORES+ Named Entity Copying Study

This study measures how much of the translation quality scores (BLEU and chrF++) on the FLORES+ benchmark comes from models copying named entities verbatim from the English source instead of translating them into the target language. By extracting named entities using a transformer-based spaCy model (`en_core_web_trf`), creating pseudo-translations consisting strictly of copied entities, and evaluating entity-masked scores, we isolate true translation quality from entity copying inflation.

## Directory Layout

```
.
├── languages.txt                 # Target FLORES+ language codes
├── requirements.txt              # Project dependencies
├── README.md                     # Main documentation
├── kaggle_run.md                 # Execution guide for Kaggle GPU environments
├── src/
│   ├── extract_entities.py       # Re-extract NEs using spaCy en_core_web_trf
│   ├── masking.py                # Core entity masking and score calculation module
│   ├── run_copy_baseline.py      # Evaluates verbatim source copy baseline
│   ├── run_entities_only_baseline.py # Evaluates entities-only pseudo-translation baseline
│   ├── compare_baselines.py     # Merges and analyzes full-copy vs entities-only baselines
│   ├── plots.py                  # Generates figures (chrF++ by script, tier, entity share)
│   ├── resource_tiers.py        # Centralized resource tier mapping
│   └── run_translation.py        # NLLB translation script (for Kaggle GPU)
├── tests/
│   └── test_masking.py           # Unit test suite for masking module
├── results/                      # Output CSVs, JSONL entities, and results README
└── figures/                      # Visualization plots
```

## Setup Instructions

1. **Environment Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download spaCy Transformer Model:**
   ```bash
   python -m spacy download en_core_web_trf
   ```

3. **Authentication:**
   Export your Hugging Face Access Token to access the FLORES+ dataset (`openlanguagedata/flores_plus`):
   ```bash
   export HF_TOKEN="your_hf_token_here"  # On Windows PowerShell: $env:HF_TOKEN="your_hf_token_here"
   ```

## Reproduction Sequence

Run the following commands in order to reproduce every result and figure:

```bash
# 1. Extract named entities from FLORES+ eng_Latn devtest
python src/extract_entities.py --output results/english_entities_trf.jsonl --skip-if-exists

# 2. Run unit tests for masking module
pytest tests/ -q

# 3. Evaluate full English copy baseline
python src/run_copy_baseline.py --output copy_baseline_all_languages.csv

# 4. Evaluate entities-only baseline for all modes (names, numeric, all)
python src/run_entities_only_baseline.py --entity-types all-modes

# 5. Merge baselines and compute entity share metrics
python src/compare_baselines.py --entities-csv results/entities_only_baseline_names.csv --output results/baseline_comparison.csv

# 6. Generate IEEE-formatted visualization figures
python src/plots.py --comparison-csv results/baseline_comparison.csv
```
