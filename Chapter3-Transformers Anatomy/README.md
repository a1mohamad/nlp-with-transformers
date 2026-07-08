# Chapter 3 — Transformer Anatomy

This section contains annotated notebooks for **Chapter 3: Transformer Anatomy** from the educational *NLP with Transformers* learning repo.

The goal of this chapter is to understand what happens inside encoder-style transformer models instead of only calling high-level Hugging Face APIs. The notebooks move from pretrained model inspection to manually implementing the main transformer components in PyTorch.

## Contents

```text
Chapter3-Transformers Anatomy/
├── README.md
├── requirements-ch03.txt
├── attention-architecture-annotated.ipynb
├── transformers-encoder-annotated.ipynb
├── output_preservation_check.json
└── original/
    ├── attention-architecture.ipynb
    └── transformers-encoder.ipynb
```

## Notebooks

### 1. `attention-architecture-annotated.ipynb`

This notebook focuses on the mechanics of **self-attention**.

Main topics:

- loading a pretrained BERT tokenizer and model
- inspecting the model architecture
- converting text into token IDs
- creating token embeddings
- computing scaled dot-product attention manually
- applying softmax to attention scores
- building a single attention head
- building multi-head attention
- collecting attention tensors from a Hugging Face model
- preparing tokens for optional BertViz attention visualization

This notebook is useful for understanding how attention turns a sequence of token vectors into contextualized representations.

### 2. `transformers-encoder-annotated.ipynb`

This notebook builds a simplified **Transformer encoder** from smaller PyTorch modules.

Main topics:

- loading DistilBERT configuration and tokenizer
- counting pretrained model parameters
- implementing scaled dot-product attention
- implementing an attention head
- implementing multi-head attention
- implementing the feed-forward network
- implementing encoder layers with residual connections and layer normalization
- implementing token and positional embeddings
- stacking encoder layers
- adding a sequence-classification head
- building and applying attention masks

This notebook is useful for connecting the mathematical transformer architecture to actual PyTorch code.

## Why outputs are preserved

The notebook outputs are intentionally kept. They are part of the learning material and show:

- model architecture printouts
- tensor shapes
- parameter counts
- embedding layer structures
- attention matrices and masks
- intermediate results from the custom implementation

Do not clear notebook outputs when committing educational versions of these notebooks unless there is a specific reason to do so.

## Recommended setup

Create and activate a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# or
.venv\Scripts\activate        # Windows
```

Then install the chapter requirements:

```bash
pip install -r requirements-ch03.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

## Notes on model downloads

These notebooks load pretrained checkpoints such as:

- `bert-base-uncased`
- `distilbert-base-uncased`

The first run may download model and tokenizer files into your local Hugging Face cache. These downloaded model files should not be committed to Git.

Recommended `.gitignore` patterns for model artifacts:

```gitignore
.cache/
hf_cache/
huggingface/
**/checkpoint-*/
*.safetensors
*.bin
*.pt
*.pth
*.onnx
```

## Suggested study order

1. Start with `attention-architecture-annotated.ipynb`.
2. Understand token embeddings and scaled dot-product attention.
3. Study the custom `AttentionHead` and `MultiHeadAttention` classes.
4. Move to `transformers-encoder-annotated.ipynb`.
5. Follow the encoder implementation from embeddings to classification head.
6. Re-run selected cells and compare the shapes with the preserved outputs.

## Learning objective

After completing this chapter, you should be able to explain how an encoder-style transformer processes text:

```text
text
→ token IDs
→ token embeddings + positional embeddings
→ multi-head self-attention
→ feed-forward layers
→ stacked encoder layers
→ task-specific output head
```

This chapter is the bridge between using pretrained models and understanding how transformer architecture is implemented internally.
