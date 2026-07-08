# Chapter 2 — Text Classification with Transformers

This section documents an educational, output-preserving implementation of **Chapter 2: Text Classification** from *Natural Language Processing with Transformers*. The chapter builds a practical emotion classifier for short English text using the Hugging Face ecosystem.

The workflow starts with dataset inspection, moves through exploratory data analysis and tokenization, then compares two modeling strategies:

1. **Feature extraction** with a pretrained DistilBERT encoder and a scikit-learn classifier.
2. **End-to-end fine-tuning** with `AutoModelForSequenceClassification` and the Hugging Face `Trainer`.

The annotated notebooks are designed for study, revision, and GitHub portfolio presentation. All original notebook outputs are intentionally preserved because the outputs show the actual dataset structure, tokenized inputs, training logs, metrics, plots, and inference results.

---

## Notebook order

Use the notebooks in this order:

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `emotions-get_familiar-annotated.ipynb` | Learn how Hugging Face `DatasetDict` objects work: splits, rows, columns, feature metadata, slicing, and loading local CSV files. |
| 2 | `emotions-EDA-annotated.ipynb` | Convert the dataset to Pandas format, inspect readable labels, plot class distribution, and analyze text length by emotion class. |
| 3 | `emotions-tokenization-annotated.ipynb` | Move from character/word tokenization intuition to real DistilBERT subword tokenization with `input_ids` and `attention_mask`. |
| 4 | `emotions-models-annotated.ipynb` | Extract hidden states, train a logistic regression baseline, fine-tune DistilBERT, evaluate with metrics/confusion matrices, perform error analysis, and run inference. |

---

## Project structure

```text
02_text_classification/
├── README.md
├── emotions-get_familiar-annotated.ipynb
├── emotions-EDA-annotated.ipynb
├── emotions-tokenization-annotated.ipynb
└── emotions-models-annotated.ipynb
```

The ZIP package also contains an `original/` folder with the unmodified uploaded notebooks for comparison.

---

## Learning objectives

By the end of this chapter section, you should be comfortable with:

- Loading public datasets from the Hugging Face Hub with `load_dataset()`.
- Understanding `DatasetDict`, dataset splits, column names, and feature metadata.
- Switching a dataset into Pandas format for quick analysis.
- Mapping integer labels to human-readable class names.
- Checking class imbalance and text-length distributions before modeling.
- Understanding why tokenization is required before text can enter a transformer.
- Comparing character, word, one-hot, and subword tokenization.
- Using `AutoTokenizer` to produce `input_ids` and `attention_mask`.
- Using `AutoModel` as a frozen feature extractor.
- Training a classical machine-learning baseline on transformer embeddings.
- Fine-tuning `AutoModelForSequenceClassification` with `Trainer`.
- Evaluating with accuracy, weighted F1, confusion matrices, and per-example losses.
- Using a fine-tuned model with the Hugging Face `pipeline()` API.

---

## Main concepts

### Text classification

Text classification assigns one label to an input text. In this chapter, each message is classified into one of six emotion categories:

```text
sadness, joy, love, anger, fear, surprise
```

The task is more specific than binary sentiment analysis because the model must choose among several emotional states rather than only positive/negative sentiment.

### Tokenization

Transformers cannot consume raw strings directly. A tokenizer converts text into integer token IDs and attention masks. DistilBERT uses a subword tokenizer, which helps handle uncommon words by splitting them into smaller pieces instead of treating every unseen word as completely unknown.

### Feature extraction vs fine-tuning

The modeling notebook demonstrates two approaches:

| Approach | What changes during training? | Why it matters |
|---|---|---|
| Frozen feature extraction | Only the scikit-learn classifier is trained. DistilBERT remains unchanged. | Fast baseline; useful when compute is limited. |
| Fine-tuning | DistilBERT and the classification head are updated for the emotion task. | Usually stronger performance; more expensive and creates model checkpoints. |

### Error analysis

The chapter does not stop after one score. It inspects confusion matrices and examples with high loss. This is an important professional habit because aggregate metrics do not explain *why* the model makes mistakes.

---

## Environment

A typical environment for these notebooks:

```bash
pip install transformers datasets huggingface_hub accelerate evaluate scikit-learn pandas matplotlib torch umap-learn
```

For local GPU training, install the PyTorch build that matches your CUDA setup from the official PyTorch installation page.

The fine-tuning notebook can run on CPU, but it is much more practical on a GPU. The official book repository also recommends GPU-backed environments for most chapters.

---

## Running the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open the notebooks in this order:

```text
emotions-get_familiar-annotated.ipynb
emotions-EDA-annotated.ipynb
emotions-tokenization-annotated.ipynb
emotions-models-annotated.ipynb
```

If you rerun the modeling notebook, training outputs may differ slightly because of hardware, package versions, randomness, and Hugging Face API changes.

---

## Hugging Face Hub notes

The modeling notebook includes:

```python
from huggingface_hub import notebook_login
notebook_login()
```

and later:

```python
trainer.push_to_hub(...)
```

These cells require a Hugging Face account and token. If you do not want to upload the model, you can skip the login and push cells, or set `push_to_hub=False` inside `TrainingArguments`.

---

## Git and model artifact policy

Do **not** commit generated model weights or training checkpoints to a normal GitHub repository. The following files can become very large:

```text
model.safetensors
pytorch_model.bin
optimizer.pt
scheduler.pt
checkpoint-*/
runs/
wandb/
```

For an educational GitHub repo, commit notebooks, scripts, README files, and small configuration files. Store trained models on the Hugging Face Hub, Git LFS, GitHub Releases, or external storage.

A useful `.gitignore` pattern is:

```gitignore
# Hugging Face training outputs
**/checkpoint-*/
**/model.safetensors
**/pytorch_model*.bin
**/optimizer.pt
**/scheduler.pt
**/rng_state*.pth
**/trainer_state.json
**/training_args.bin
runs/
wandb/
artifacts/
outputs/
models/
```

---

## Output preservation policy

The annotated notebooks preserve the original outputs. This is intentional.

Notebook outputs are part of the learning material in this repository because they show:

- dataset structure,
- example rows,
- label mappings,
- plots,
- tokenized dictionaries,
- tensor shapes,
- training logs,
- metrics,
- confusion matrices,
- high-loss and low-loss examples,
- inference results.

When updating these notebooks, do not clear outputs unless the purpose is to create a separate clean execution version.
