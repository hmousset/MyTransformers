# How to Run MyTransformers

This guide explains the setup needed before running training, the expected dataset format, common launch commands, inference, and checkpoint conversion.

## 1. Create an Environment

Use a Python 3.8+ environment. Recommended: [uv](https://github.com/astral-sh/uv)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

If PyTorch needs a specific CUDA wheel on your machine, install that PyTorch build first, then run `uv pip install -e .`.

## 2. Create `paths.json`

The project imports `common.paths` during startup. That file requires `paths.json` in the repository root.

Use an empty file when you provide every path explicitly:

```json
{}
```

Use named entries when you want shorter scripts:

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

With this example:

- `--tokenizer-name llama3` resolves `tokenizer_llama3`.
- `--model-name llama3 --variant 8b` resolves `model_llama3_8b`.
- `--huggingface --model-name llama3 --variant 8b` resolves `huggingface_llama3_8b`.
- `--train-dataset-name sft` resolves `train_dataset_sft`.
- `--eval-dataset-name sft` resolves `eval_dataset_sft`.

`paths.json` is ignored by Git so local machine paths do not get committed.

## 3. Prepare Data

For SFT, use JSONL with input and output fields:

```jsonl
{"input": "Explain what LoRA does.", "output": "LoRA trains small low-rank adapters while keeping most base weights frozen."}
{"input": "What is DNA methylation?", "output": "DNA methylation is an epigenetic mark that can regulate gene expression."}
```

For pretraining, each line can be plain text:

```text
This is one pretraining document.
This is another pretraining document.
```

Or JSONL with an `input` field:

```jsonl
{"input": "This is one pretraining document."}
{"input": "This is another pretraining document."}
```

Useful dataset arguments:

- `--mode sft` requires both input and output text.
- `--mode pretrain` trains on input text and optional output text concatenated together.
- `--dataset-input-field` defaults to `input`.
- `--dataset-output-field` defaults to `output`.
- `--max-src-len` limits the source/input side.
- `--max-len` limits the full tokenized sequence.
- `--skip-eval` is required when you do not pass an eval dataset.

## 4. Run a Hugging Face Model

This is the simplest smoke test for a local Hugging Face model path:

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

Add checkpoint saving:

```bash
  --output-path output \
  --save-interval 100
```

Add LoRA:

```bash
  --use-lora \
  --replace-modules q_proj k_proj v_proj o_proj \
  --lora-rank 8 \
  --lora-scaler 16 \
  --save-trainable \
  --output-path output \
  --save-interval 100
```

For other Hugging Face architectures, inspect module names with Python or `print(model)` and set `--replace-modules` to the linear layers you want to adapt.

## 5. Run a Local MyTransformers Model

Local models use the project registry. Supported local names include `llama`, `llama1`, `llama2`, `llama3`, `gemma`, and multimodal DNA variants registered in `model/*`.

Example local Llama LoRA run:

```bash
deepspeed --num_gpus 1 train/u_train.py \
  --model-name llama3 \
  --variant 8b \
  --tokenizer-name llama3 \
  --tokenizer-path /path/to/tokenizer.model \
  --ckpt-path /path/to/model.ckpt \
  --from-pretrained \
  --train-dataset-path /path/to/train.jsonl \
  --dataset-class-name iterable \
  --mode sft \
  --max-len 1024 \
  --max-src-len 768 \
  --batch-size-per-gpu 1 \
  --gradient-accumulation-steps 8 \
  --train-iters 100 \
  --show-avg-loss-step 1 \
  --lr 1e-4 \
  --bf16 \
  --device cuda \
  --zero-stage 2 \
  --use-lora \
  --replace-modules wq wk wv wo \
  --lora-rank 8 \
  --lora-scaler 16 \
  --save-trainable \
  --output-path output \
  --save-interval 100 \
  --skip-eval \
  --experiment-name local_lora_sft
```

If you use `paths.json`, replace explicit paths with names:

```bash
  --model-name llama3 \
  --variant 8b \
  --tokenizer-name llama3 \
  --train-dataset-name sft
```

## 6. Evaluation

Pass an eval dataset by path:

```bash
  --eval-dataset-path /path/to/eval.jsonl \
  --eval-interval 500 \
  --eval-max-len 1024 \
  --eval-max-src-len 768 \
  --eval-batch-size-per-gpu 1
```

Or by name:

```bash
  --eval-dataset-name sft
```

If no eval dataset is available, use:

```bash
  --skip-eval
```

## 7. Logging

Rank-0 logs are written under `~/MyTransformers_log` by default. Override the folder:

```bash
LOG_FOLDER=/tmp/mytransformers_logs deepspeed --num_gpus 1 train/u_train.py ...
```

TensorBoard:

```bash
  --tensorboard \
  --tb-log-dir /path/to/tensorboard
```

Weights & Biases:

```bash
  --wandb \
  --wandb-dir /path/to/wandb \
  --wandb-cache-dir /path/to/wandb-cache \
  --wandb-project MyTransformers
```

## 8. Inference

[scripts/run.py](../scripts/run.py) loads local MyTransformers model checkpoints and optionally LoRA/fine-tuned checkpoint deltas.

Single prompt:

```bash
python scripts/run.py \
  --tokenizer /path/to/tokenizer.model \
  --pretrained_ckpt /path/to/base.ckpt \
  --ckpt /path/to/fine_tuned_or_lora.ckpt \
  --model_name llama3 \
  --variant 8b \
  --prompt "Explain LoRA in one paragraph." \
  --output_len 128 \
  --device cuda
```

Interactive loop:

```bash
python scripts/run.py \
  --tokenizer /path/to/tokenizer.model \
  --pretrained_ckpt /path/to/base.ckpt \
  --model_name llama3 \
  --variant 8b \
  --run_loop \
  --device cuda
```

Dataset inference expects `csv`, `xlsx`, `json`, or `jsonl` with `input` and `output` fields:

```bash
python scripts/run.py \
  --tokenizer /path/to/tokenizer.model \
  --pretrained_ckpt /path/to/base.ckpt \
  --ckpt /path/to/fine_tuned_or_lora.ckpt \
  --model_name llama3 \
  --variant 8b \
  --dataset_path /path/to/eval.jsonl \
  --result_path output/inference \
  --batch_size 8 \
  --output_len 128 \
  --device cuda
```

## 9. Checkpoint Conversion

Use [tools/convert_checkpoint.py](../tools/convert_checkpoint.py) to merge pipeline-parallel checkpoint shards or merge LoRA weights into a normal checkpoint.

Example:

```bash
python tools/convert_checkpoint.py \
  --model_path output/local_lora_sft/final.ckpt \
  --pretrained_model_path /path/to/base.ckpt \
  --save_model_dir output/merged \
  --save_name merged_full \
  --tokenizer /path/to/tokenizer.model \
  --model_name llama3 \
  --variant 8b
```

Add `--pipeline_model` when converting a DeepSpeed pipeline checkpoint directory. Add `--not_merge_lora` if you want to keep LoRA weights separate.

## 10. Common Issues

`FileNotFoundError: Please config paths in .../paths.json`

Create `paths.json` in the repository root. `{}` is valid when paths are passed explicitly.

`args.eval_dataset_path can not be None`

Pass `--skip-eval` or provide `--eval-dataset-path` / `--eval-dataset-name`.

`Model saving requires an output path`

When using `--save-interval` or `--save-epoch`, also pass `--output-path`.

LoRA has no effect with `--use-lora-plus`

Add `--diy-optimizer`; LoRA+ needs the custom optimizer path.

Wrong LoRA target modules

Set `--replace-modules` to substrings that actually appear in the model's linear layer names.

Out of memory

Reduce `--batch-size-per-gpu`, increase `--gradient-accumulation-steps`, enable `--bf16` or `--fp16`, use `--zero-stage 2` or `--zero-stage 3`, train LoRA only, or lower `--max-len`.

