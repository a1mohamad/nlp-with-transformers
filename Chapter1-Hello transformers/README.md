# Chapter 1 — Hello Transformers

This section is the starting point of the **Build NLP with Transformers** learning repository. It introduces the practical side of Transformer-based NLP using the Hugging Face `pipeline()` API.

The notebook is intentionally simple: one customer complaint is passed through several pretrained models so you can see how different NLP tasks behave without writing training loops, tokenization code, or model classes manually.

The annotated notebook keeps the original outputs. This matters because Chapter 1 is mostly about seeing what pretrained pipelines return: labels, scores, entity spans, answer spans, generated summaries, translations, warnings, and generated text.

---

## What this chapter teaches

By the end of this notebook, you should understand how to use pretrained Transformer models for:

| Task | What it does |
|---|---|
| Text classification | Predicts a label such as positive or negative sentiment |
| Named entity recognition | Detects entities such as people, organizations, and locations |
| Question answering | Extracts an answer span from a context paragraph |
| Summarization | Compresses a longer text into a shorter version |
| Translation | Converts text from one language to another |
| Text generation | Continues a prompt with generated text |

The main concept is the pipeline abstraction:

```text
input text → tokenizer → pretrained Transformer model → postprocessing → final result
```

A pipeline lets you focus on the task first. Later chapters go deeper into what happens inside the tokenizer, the model, the training process, and the evaluation workflow.

---

## Files in this section

```text
01_hello_transformers/
├── README.md
└── hello-transformers-annotated.ipynb
```

### `hello-transformers-annotated.ipynb`

An educational version of the original chapter notebook with:

- a full introductory markdown cell,
- markdown explanations before every code cell,
- line-level comments for important code,
- preserved original outputs,
- the original progression of tasks from the chapter,
- a corrected English-to-German model identifier in the translation setup cell.

---

## Pipelines used in the notebook

| Task | Pipeline call |
|---|---|
| Text classification | `pipeline("text-classification")` |
| Named entity recognition | `pipeline("ner", aggregation_strategy="simple")` |
| Question answering | `pipeline("question-answering")` |
| Summarization | `pipeline("summarization")` |
| Translation | `pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")` |
| Text generation | `pipeline("text-generation")` |

Some cells intentionally rely on Hugging Face defaults, which is useful for a first chapter demonstration because it shows the warning messages and default-model behavior. In production code, explicit model names and revisions are recommended.

---

## How to run

From the repository root:

```bash
pip install -r requirements.txt
jupyter lab
```

Then open:

```text
notebooks/01_hello_transformers/hello-transformers-annotated.ipynb
```

The first run may take extra time because the models need to be downloaded from the Hugging Face Hub. After that, they are cached locally.

---

## Important notes

- Internet access is required the first time each model is downloaded.
- CPU is enough for this notebook, but GPU will make larger model inference faster.
- Text generation may vary across reruns depending on decoding behavior and library versions.
- Saved notebook outputs are preserved so the file is still educational before rerunning.
- The goal of this section is exploration, not fine-tuning or production deployment.
- If a model download fails, check the model name, your internet connection, and your installed `transformers` version.

---

## Suggested experiments

Try these after reading the notebook once:

1. Replace the Amazon complaint with your own text.
2. Ask different questions in the QA cell.
3. Compare short and long summaries.
4. Translate into another language using a different translation model.
5. Change the generation prompt and compare the output.
6. Specify exact model names for every pipeline and compare warnings.

---

## Key takeaway

Chapter 1 shows the power of transfer learning in NLP. With a few lines of code, you can use pretrained Transformer models for many practical language tasks. The rest of the learning path explains how these systems work, how to adapt them to custom data, and how to make them efficient enough for real applications.
