# MyTransformers

MyTransformers is a research training codebase for language models, multimodal omics models, and many LoRA-style parameter-efficient fine-tuning methods. It provides local PyTorch model implementations, Hugging Face model loading, DeepSpeed distributed training, sequence and pipeline parallelism, flexible dataset wrappers, and a registry system for models, tokenizers, datasets, and paths.

## Documentation

- [How to run the code](docs/running.md)
- [Architecture guide](docs/architecture.md)
- [Training package notes](train/readme.md)
- [Common utilities notes](common/readme.md)
- [Script and argument notes](scripts/readme.md)

The original Chinese PDF is still present as `MyTransformers_document.pdf`; the Markdown guides above are the English documentation you should use day to day.

## Requirements

The project targets Python 3.8+ and GPU training with PyTorch and DeepSpeed.

Core packages are listed in [requirements.txt](requirements.txt):

- `torch`
- `deepspeed`
- `transformers`
- `liger_kernel`
- `bitsandbytes`
- `vllm`
- `wandb`
- `sentencepiece`
- additional utility packages used by the training and inference scripts

Install in editable mode from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install the CUDA-compatible PyTorch build for your machine before or during this step if the default package index does not match your GPU/CUDA setup.

## Required Local Paths

The `common` package loads `paths.json` during import. Create it in the repository root before running training or importing most project modules.

For fully explicit command-line paths, an empty file is enough:

```json
{}
```

For named paths, use this shape:

```json
{
  "tokenizer": {
    "llama3": "/models/llama3/tokenizer.model"
  },
  "model": {
    "llama3_8b": "/models/llama3/consolidated.pth"
  },
  "huggingface": {
    "llama3_8b": "/models/hf/Meta-Llama-3-8B"
  },
  "train_dataset": {
    "sft": "/data/train.jsonl"
  },
  "eval_dataset": {
    "sft": "/data/eval.jsonl"
  }
}
```

Then `--model-name llama3 --variant 8b --tokenizer-name llama3 --train-dataset-name sft` can resolve paths automatically.

## Quick Training Example

For a small Hugging Face SFT smoke run:

```bash
deepspeed --num_gpus 1 train/u_train.py \
  --huggingface \
  --model-name-or-path /path/to/huggingface-model \
  --train-dataset-path /path/to/train.jsonl \
  --dataset-class-name iterable \
  --mode sft \
  --dataset-input-field input \
  --dataset-output-field output \
  --max-len 1024 \
  --max-src-len 768 \
  --batch-size-per-gpu 1 \
  --gradient-accumulation-steps 8 \
  --train-iters 100 \
  --show-avg-loss-step 1 \
  --lr 2e-5 \
  --bf16 \
  --device cuda \
  --zero-stage 2 \
  --skip-eval \
  --experiment-name smoke_hf_sft
```

For LoRA, add:

```bash
  --use-lora \
  --replace-modules q_proj k_proj v_proj o_proj \
  --lora-rank 8 \
  --lora-scaler 16 \
  --save-trainable \
  --output-path output
```

See [docs/running.md](docs/running.md) for local model training, dataset formats, inference, checkpoint conversion, logging, and common failures.

## Supported Features

1. DeepSpeed distributed training with data parallelism, ZeRO, pipeline parallelism, sequence parallelism, and multi-node launch support.
2. Hugging Face model and tokenizer loading, plus local PyTorch implementations for Llama, Gemma, DNA Hyena, and DNABERT-related models.
3. More than 20 LoRA and LoRA-variant implementations, including DoRA, QLoRA-style paths, AdaLoRA, GoRA, ReLoRA, PiSSA, LoRA-GA, and others.
4. Flexible optimizer and learning-rate scheduler setup, including custom parameter groups.
5. Multiple attention implementations, including SDPA-style Hugging Face fallback and project attention variants such as FlashAttention/Ulysses paths where configured.
6. Several dataset classes, including normal, iterable, concatenated, packed, and multimodal DNA datasets.
7. A registry mechanism for models, training wrappers, pipeline wrappers, tokenizers, datasets, and named filesystem paths.

## Project Outputs

- [NeurIPS 2025] GoRA: Gradient-driven Adaptive Low Rank Adaptation
- [EMNLP 2025 Findings] Biology-Instructions: A Dataset and Benchmark for Multi-Omics Sequence Understanding Capability of Large Language Models
- [ICLR 2026] Gradient Intrinsic Dimensionality Alignment: Narrowing The Gap Between Low-Rank Adaptation and Full Fine-Tuning
- [ICLR 2026] E2LoRA: Efficient and Effective Low-Rank Adaptation with Entropy-Guided Adaptive Sharing
- [Under Review] Rethinking Multi-Omics LLMs from the Perspective of Omics-Encoding
- [Under Review] A Unified Study of LoRA Variants: Taxonomy, Review, Codebase, and Empirical Evaluation

