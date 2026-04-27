# Scripts and Training Arguments

The scripts in this directory are examples. Replace placeholder paths before running them.

## Main Training Flags

- Full fine-tuning vs LoRA is controlled mainly by `--use-lora` or LoRA-variant flags such as `--use-lora-plus`, `--use-dora`, `--use-plora`, and `--use-lora-fa`.
- `--use-dora`, `--use-plora`, and `--use-lora-fa` still require `--use-lora`.
- When using LoRA, the default local-model target modules come from `model_config.lora_layers`. Override them with `--replace-modules`, for example `--replace-modules q_proj k_proj v_proj o_proj` for many Hugging Face decoder models.
- `--use-lora-plus` requires `--diy-optimizer`; otherwise it behaves like regular LoRA.
- For full fine-tuning, use `--disable-list` to freeze parameters whose names contain any listed substring.
- Use `--enable-list` to train only parameters whose names contain listed substrings. This is the inverse of `--disable-list`.
- Switch between pretraining and supervised fine-tuning with `--mode pretrain` or `--mode sft`. The main difference is how dataset input/output fields are processed.
- Set training length with either `--epochs` or `--train-iters`, not both.
- Select a local model with `--model-name` and `--variant`.
- The default log folder is `~/MyTransformers_log`. Override it with `LOG_FOLDER=/path/to/logs`.

## Dataset Flags

- `--dataset-class-name normal`: eager map-style JSONL/text dataset.
- `--dataset-class-name iterable`: lazy iterable JSONL/text dataset.
- `--dataset-class-name concat` or `concat_iterable`: combine multiple dataset files.
- `--batching-stretegy padding`: pad every sample to `--max-len`.
- `--batching-stretegy packing`: pack tokenized samples into chunks of `--max-len`.
- `--dataset-input-field` and `--dataset-output-field` select JSON object fields. Defaults are `input` and `output`.
- For concatenated datasets, pass space-separated weights, for example `--dataset-weights 2 1 1`.

## Examples

Run a Hugging Face SFT job:

```bash
deepspeed --num_gpus 1 train/u_train.py \
  --huggingface \
  --model-name-or-path /path/to/huggingface-model \
  --train-dataset-path /path/to/train.jsonl \
  --dataset-class-name iterable \
  --mode sft \
  --max-len 1024 \
  --max-src-len 768 \
  --batch-size-per-gpu 1 \
  --gradient-accumulation-steps 8 \
  --train-iters 100 \
  --bf16 \
  --zero-stage 2 \
  --skip-eval \
  --experiment-name smoke_hf_sft
```

Run LoRA on a local Llama model:

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
  --bf16 \
  --zero-stage 2 \
  --use-lora \
  --replace-modules wq wk wv wo \
  --lora-rank 8 \
  --lora-scaler 16 \
  --save-trainable \
  --output-path output \
  --skip-eval \
  --experiment-name local_lora_sft
```

Run local-model inference with [run.py](run.py):

```bash
python scripts/run.py \
  --tokenizer /path/to/tokenizer.model \
  --pretrained_ckpt /path/to/base.ckpt \
  --ckpt /path/to/lora_or_finetuned.ckpt \
  --model_name llama3 \
  --variant 8b \
  --prompt "Write a short answer about DNA methylation." \
  --output_len 128 \
  --device cuda
```
