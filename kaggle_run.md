# Kaggle Notebook – Run NLLB Translation Pipeline

This document gives exact copy-paste steps to run `src/run_translation.py`
inside a Kaggle notebook with a free T4 GPU and persistent output.

---

## Prerequisites

1. You must have a [Kaggle account](https://www.kaggle.com).
2. You must have accepted the FLORES+ dataset terms at:
   <https://huggingface.co/datasets/openlanguagedata/flores_plus>
3. You need a Hugging Face **User Access Token** (read scope is enough).
   Create one at <https://huggingface.co/settings/tokens>.

---

## Step 1 – Store your HF token as a Kaggle Secret

1. Go to <https://www.kaggle.com/settings> → **API** → **Secrets**.
2. Click **Add a new secret**.
3. Name: `HF_TOKEN`  |  Value: your token (e.g. `hf_xxxxx…`)
4. Click **Save**.

---

## Step 2 – Create a new Kaggle notebook

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Under **Settings** → **Accelerator**, choose **GPU T4 × 1**.
3. Under **Settings** → **Persistence**, choose **Files** (so `results/` survives between sessions).
4. Under **Settings** → **Internet**, make sure **Internet On** is enabled.

---

## Step 3 – Clone this repository

Paste into the first code cell and run:

```python
import subprocess, os

# Load HF token from Kaggle Secret into environment
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# Clone repo
subprocess.run([
    "git", "clone",
    "https://github.com/theniyazkhan/nlp.git",
    "/kaggle/working/nlp"
], check=True)
```

---

## Step 4 – Install dependencies

```python
subprocess.run([
    "pip", "install", "-q",
    "datasets", "transformers", "accelerate", "sentencepiece",
    "sacrebleu", "tqdm", "huggingface_hub"
], check=True)
```

> **Optional – 8-bit quantisation** (saves ~50% VRAM, slightly slower):
> ```python
> subprocess.run(["pip", "install", "-q", "bitsandbytes"], check=True)
> ```

---

## Step 5 – Run the translation script

```python
os.chdir("/kaggle/working/nlp")

result = subprocess.run([
    "python", "src/run_translation.py",
    "--model", "facebook/nllb-200-distilled-600M",
    "--languages-file", "languages.txt",
    "--output-dir", "results/translations",
    "--batch-size", "32",
    # "--load-8bit",   # uncomment if you installed bitsandbytes
], capture_output=False)
```

Translation outputs land in:
```
results/translations/facebook_nllb-200-distilled-600M/<lang_code>.jsonl
```

Each `.jsonl` file has 1012 lines of `{"sentence_id": …, "source": …, "hypothesis": …}`.

---

## Step 6 – Enable persistent output for downloading

Kaggle only persists `/kaggle/working/`. The results directory is already there.

After the run finishes:
1. In the notebook sidebar, click **Output** → **Expand output**.
2. Locate `results/translations/` → click the `⋮` menu → **Download**.
3. Alternatively, commit the notebook version to save outputs to your Kaggle account.

---

## Step 7 – (Optional) Push results back to GitHub

```python
subprocess.run(["git", "config", "user.name", "kaggle-runner"], check=True)
subprocess.run(["git", "config", "user.email", "kaggle@example.com"], check=True)
subprocess.run(["git", "add", "results/"], check=True)
subprocess.run(["git", "commit", "-m", "Add NLLB translation outputs"], check=True)
subprocess.run([
    "git", "push",
    f"https://theniyazkhan:{os.environ['HF_TOKEN']}@github.com/theniyazkhan/nlp.git",
    "main"
], check=True)
```

> ⚠️ The HF token is used here only as a GitHub PAT placeholder.
> Replace with a **GitHub Personal Access Token** if you need to push.

---

## Resumability

The script automatically skips any language whose `.jsonl` file already has
exactly 1012 lines. If the notebook times out mid-run, simply re-run Step 5
and it will continue from where it left off.

---

## Expected Runtime

| Model | Batch size | T4 GPU | ~time per language |
|---|---|---|---|
| nllb-200-distilled-600M | 32 | ✓ | ~2–4 min |
| nllb-200-distilled-1.3B | 16 | ✓ | ~5–8 min |
| nllb-200-3.3B | 8 | ✓ (8-bit) | ~12–18 min |

32 languages × 3 min ≈ **~1.5 hours** for the full run on the 600M model.
