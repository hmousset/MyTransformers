"""
LoRA fine-tuning for LLMs without DeepSpeed.

Callable as a Python function from ExperimentalGrow:
    from train.train_lora_llm import train_lora_llm, LoRALLMConfig

Or as a CLI script:
    python train/train_lora_llm.py --model-name-or-path ... --train-dataset-path ...
"""
from __future__ import annotations

import os
import json
import logging
import dataclasses
from dataclasses import dataclass, field
from argparse import Namespace
from typing import Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.utils import is_liger_kernel_available
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import DataLoader

from common.lora_modules import setup_lora, prepare_lora
from common.utils.params_manager import set_up_trainable_param
from common.utils import print_rank_0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoRALLMConfig:
    """All parameters for a single LoRA-LLM training run."""

    # ---- model ----
    model_name_or_path: str = ""

    # ---- dataset ----
    train_dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    dataset_input_field: str = "input"
    dataset_output_field: str = "output"
    max_len: int = 1024

    # ---- training ----
    experiment_name: str = "lora_llm"
    output_path: Optional[str] = None
    epochs: Optional[int] = 3
    train_iters: Optional[int] = None
    batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    eval_interval: int = 500
    save_interval: Optional[int] = None
    wandb: bool = False
    activation_checkpoint: bool = False

    # ---- LoRA core ----
    use_lora: bool = True
    lora_rank: int = 8
    lora_scaler: int = 16
    lora_dropout: Optional[float] = None
    replace_modules: Optional[List[str]] = None
    run_lora_in_fp32: bool = False
    std_normalize_lora: bool = False
    lora_reset_weight: bool = False
    weight_a_init_method: str = "kaiming"
    weight_b_init_method: str = "zero"
    weight_c_init_method: Optional[str] = None
    enable_list: Optional[List[str]] = None
    disable_list: Optional[List[str]] = None
    params_to_save: Optional[List[str]] = None

    # ---- LoRA variant flags (all False by default) ----
    use_dora: bool = False
    use_hira: bool = False
    use_mos_lora: bool = False
    weight_ab_mixer_init_method: Optional[str] = None
    use_me_lora: bool = False
    me_lora_n_split: int = 2
    me_lora_forward_method: str = "for"
    use_lora_ga: bool = False
    use_lora_one: bool = False
    use_rslora: bool = False
    use_pissa: bool = False
    pissa_n_iters: int = 0
    pissa_keep_init_weights: bool = False
    use_olora: bool = False
    use_vera: bool = False
    vera_init_unique_weights: bool = False
    use_sharelora: bool = False
    sharelora_share_part: str = "AB"
    use_tied_lora: bool = False
    use_randlora: bool = False
    randlora_use_sparse: bool = False
    randlora_sparse_factor: float = 3.0
    use_lora_pro: bool = False
    use_lora_plus: bool = False
    lora_plus_scaler: int = 16
    use_nlora: bool = False
    use_dude: bool = False
    use_lora_ga_pro: bool = False
    use_loraga_pro: bool = False
    use_goat: bool = False
    use_lora_sb: bool = False
    use_gora: bool = False
    gora_n_steps: int = 32
    gora_init_method: str = "compress"
    gora_max_rank: int = 32
    gora_min_rank: int = 4
    gora_scale_by_lr: bool = False
    gora_rank_stablize: bool = False
    gora_dynamic_scaling: bool = False
    gora_importance_type: str = "union_mean"
    gora_lr: float = 5e-2
    use_increlora: bool = False
    use_mola: bool = False
    use_nora: bool = False
    use_mora: bool = False
    use_delta_lora: bool = False
    delta_lora_start_steps: Optional[int] = None
    use_adalora: bool = False
    use_plora: bool = False
    use_salora: bool = False
    use_delora: bool = False
    delora_lambda: float = 1.0
    use_eva: bool = False
    use_ralora: bool = False
    ralora_allocate_by_erank: bool = False
    ralora_disable_n_split: bool = False
    use_dralora: bool = False
    use_lora_da: bool = False
    use_milora: bool = False
    use_melora: bool = False
    use_lora_moe: bool = False
    lora_moe_aux_loss_coeff: float = 0.0
    use_rasa: bool = False
    rasa_shared_lora_rank: int = 1
    use_rasamoe: bool = False
    use_bslora: bool = False
    use_dense_lora: bool = False
    use_qlora: bool = False
    use_nzlora: bool = False
    nzlora_init_scale_a: float = 1.0
    nzlora_init_scale_b: float = 1.0
    use_relora: bool = False
    relora_steps: Optional[int] = None
    relora_reset_optimizer: bool = False
    relora_fully_reset_optimizer: bool = False
    relora_optimizer_random_pruning: float = 0.0
    relora_optimizer_magnitude_pruning: float = 0.0
    use_loda: bool = False
    use_loha: bool = False
    use_lokr: bool = False
    use_prolora: bool = False
    use_ridgelora: bool = False
    use_adalomo: bool = False
    use_aurora: bool = False
    use_lora_fa: bool = False
    use_loran: bool = False
    use_sinelora: bool = False
    use_lora_dash: bool = False

    # ---- variant-specific params ----
    plora_momentum: float = 0.9
    mora_type: str = "rope"
    nora_n_iters: int = 0
    milora_n_iters: int = 0
    init_r: int = 12
    target_r: int = 4
    delta_lora_update_ratio: float = 0.3
    lora_moe_n_experts: int = 4
    lora_moe_top_k: int = 2
    lambda_b_init_method: str = "ones"
    lambda_d_init_method: str = "ones"
    goat_scaling_type: str = "lora"
    goat_init_type: str = "default"
    goat_init_cof: float = 1.0
    goat_eta: float = 1.0
    goat_rho: float = 1.0
    lora_ga_pro_rank_stablize: bool = False
    lora_ga_pro_dynamic_scaling: bool = False
    ralora_dynamic_scaling: bool = False
    ralora_forward_method: str = "default"
    prolora_shared_rank: int = 1
    bslora_forward_method: str = "default"
    sinelora_freq: float = 1.0
    loran_freq: float = 1.0
    loran_amp: float = 1.0
    nzlora_init_scale_a: float = 1.0
    nzlora_init_scale_b: float = 1.0
    delora_lambda: float = 1.0
    lokr_k: int = 2
    lokr_decompose_weight_c: bool = False
    gora_rank_stablize: bool = False
    gora_dynamic_scaling: bool = False
    gora_importance_type: str = "union_mean"
    gora_lr: float = 5e-2
    gora_scale_by_lr: bool = False
    dash_lora_index: Optional[int] = None
    dash_lora_init_t: float = 1.0

    # ---- init steps for gradient-based LoRA variants ----
    gradient_est_n_steps: int = 1

    # ---- internal (set automatically) ----
    global_rank: int = dataclasses.field(default=0, init=False)
    local_rank: int = dataclasses.field(default=0, init=False)
    world_size: int = dataclasses.field(default=1, init=False)
    huggingface: bool = dataclasses.field(default=True, init=False)
    num_sp_stages: int = dataclasses.field(default=0, init=False)
    num_pp_stages: int = dataclasses.field(default=0, init=False)
    mode: str = dataclasses.field(default="sft", init=False)
    apply_chat_template: bool = dataclasses.field(default=False, init=False)
    meta_prompt: str = dataclasses.field(default="", init=False)
    prefix: str = dataclasses.field(default="", init=False)
    postfix: str = dataclasses.field(default="", init=False)
    rank: int = dataclasses.field(default=0, init=False)

    def __post_init__(self):
        self.default_dtype = "bf16" if self.bf16 else ("fp16" if self.fp16 else "fp32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = self.lora_rank

    def to_namespace(self) -> Namespace:
        ns = Namespace(**{k: v for k, v in dataclasses.asdict(self).items()})
        ns.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ns

    @classmethod
    def from_dict(cls, d: dict) -> "LoRALLMConfig":
        valid = {f.name for f in dataclasses.fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_hf_dataset(path: str) -> HFDataset:
    """Load a local JSONL/JSON/CSV file or a HuggingFace dataset name."""
    if os.path.exists(path):
        ext = path.rsplit(".", 1)[-1].lower()
        fmt = {"jsonl": "json", "json": "json", "csv": "csv", "tsv": "csv"}.get(ext, "json")
        return load_dataset(fmt, data_files=path, split="train")
    return load_dataset(path, split="train")


def _tokenize_sft(example, tokenizer, input_field, output_field, max_len):
    prompt = example[input_field]
    response = example[output_field]

    if tokenizer.chat_template:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer(full_text, truncation=True, max_length=max_len)
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        labels = [-100] * len(prompt_ids) + ids["input_ids"][len(prompt_ids):]
        ids["labels"] = labels[: max_len]
    else:
        src = prompt
        tgt = response + (tokenizer.eos_token or "")
        src_ids = tokenizer(src, truncation=True, max_length=max_len // 2)["input_ids"]
        tgt_ids = tokenizer(tgt, truncation=True, max_length=max_len - len(src_ids))["input_ids"]
        input_ids = src_ids + tgt_ids
        labels = [-100] * len(src_ids) + tgt_ids
        ids = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    return ids


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_lora_llm(config: LoRALLMConfig | dict) -> None:
    """
    Fine-tune an LLM with a MyTransformers LoRA variant, no DeepSpeed.

    Args:
        config: A LoRALLMConfig instance or a plain dict with the same fields.
    """
    if isinstance(config, dict):
        config = LoRALLMConfig.from_dict(config)

    set_seed(config.seed)
    args = config.to_namespace()

    # --- Load model and tokenizer ---
    print_rank_0(f"--> Loading model from {config.model_name_or_path}", 0)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.default_dtype]

    if is_liger_kernel_available():
        from train.safe_liger_kernel import AutoLigerKernelForCausalLM as _ModelCls
    else:
        _ModelCls = AutoModelForCausalLM

    model = _ModelCls.from_pretrained(
        config.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=False if config.activation_checkpoint else True,
    )
    if config.activation_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Apply LoRA ---
    if config.use_lora:
        print_rank_0(f"--> Applying LoRA (rank={config.lora_rank})", 0)
        setup_lora(model, args)
        set_up_trainable_param(model, args)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print_rank_0(f"--> Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", 0)

    # --- Load datasets ---
    assert config.train_dataset_path, "train_dataset_path is required"
    raw_train = _load_hf_dataset(config.train_dataset_path)
    raw_eval = _load_hf_dataset(config.eval_dataset_path) if config.eval_dataset_path else None

    tok_kwargs = dict(
        tokenizer=tokenizer,
        input_field=config.dataset_input_field,
        output_field=config.dataset_output_field,
        max_len=config.max_len,
    )
    train_dataset = raw_train.map(lambda ex: _tokenize_sft(ex, **tok_kwargs), remove_columns=raw_train.column_names)
    eval_dataset = raw_eval.map(lambda ex: _tokenize_sft(ex, **tok_kwargs), remove_columns=raw_eval.column_names) if raw_eval else None

    # --- prepare_lora (needed for gradient-based inits: LoRA-GA, GoRA, etc.) ---
    _needs_grad_init = any([
        config.use_lora_ga, config.use_lora_one, config.use_gora, config.use_loraga_pro,
        config.use_lora_sb, config.use_eva, config.use_ralora, config.use_dralora,
        config.use_lora_da, config.use_vera, config.use_sharelora, config.use_randlora,
        config.use_rasa, config.use_dense_lora, config.use_rasamoe, config.use_bslora,
        config.use_adalora, config.use_increlora, config.use_mola,
    ])
    if _needs_grad_init:
        print_rank_0("--> Running pre-training LoRA initialization (gradient-based)...", 0)
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        prep_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size_per_gpu,
            shuffle=True,
            collate_fn=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
        )
        prepare_lora(model, prep_loader, args)

    # --- Training arguments ---
    output_dir = os.path.join(config.output_path or ".", config.experiment_name)
    _use_gpu = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs or 1,
        max_steps=config.train_iters or -1,
        per_device_train_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16 and _use_gpu,
        fp16=config.fp16 and _use_gpu,
        use_cpu=not _use_gpu,
        save_strategy="steps" if config.save_interval else "no",
        save_steps=config.save_interval or 0,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_interval,
        logging_steps=10,
        seed=config.seed,
        report_to=["wandb"] if config.wandb else ["none"],
        run_name=config.experiment_name,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True)

    import inspect as _inspect
    _trainer_sig = _inspect.signature(Trainer.__init__).parameters
    _tok_kwarg = "processing_class" if "processing_class" in _trainer_sig else "tokenizer"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **{_tok_kwarg: tokenizer},
        data_collator=data_collator,
    )

    print_rank_0("--> Starting training", 0)
    trainer.train()

    # --- Save final checkpoint ---
    if config.output_path:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        cfg_path = os.path.join(output_dir, "lora_config.json")
        with open(cfg_path, "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2, default=str)
        print_rank_0(f"--> Saved to {output_dir}", 0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_cli() -> LoRALLMConfig:
    import argparse

    p = argparse.ArgumentParser(description="LoRA LLM training without DeepSpeed")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--train-dataset-path", required=True)
    p.add_argument("--eval-dataset-path", default=None)
    p.add_argument("--dataset-input-field", default="input")
    p.add_argument("--dataset-output-field", default="output")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--experiment-name", default="lora_llm")
    p.add_argument("--output-path", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-iters", type=int, default=None)
    p.add_argument("--batch-size-per-gpu", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--activation-checkpoint", action="store_true")

    # LoRA core
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-scaler", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--replace-modules", nargs="*", default=None)
    p.add_argument("--run-lora-in-fp32", action="store_true")
    p.add_argument("--weight-a-init-method", default="kaiming")
    p.add_argument("--weight-b-init-method", default="zero")

    # LoRA variants (flags)
    for flag in [
        "use-dora", "use-hira", "use-mos-lora", "use-me-lora",
        "use-lora-ga", "use-lora-one", "use-rslora", "use-pissa",
        "use-olora", "use-vera", "use-sharelora", "use-randlora",
        "use-lora-pro", "use-lora-plus", "use-nlora", "use-dude",
        "use-lora-ga-pro", "use-goat", "use-lora-sb", "use-gora",
        "use-increlora", "use-mola", "use-nora", "use-mora",
        "use-delta-lora", "use-adalora", "use-plora", "use-salora",
        "use-delora", "use-eva", "use-ralora", "use-dralora",
        "use-lora-da", "use-milora", "use-melora", "use-lora-moe",
        "use-rasa", "use-rasamoe", "use-bslora", "use-dense-lora",
        "use-qlora", "use-nzlora", "use-relora",
    ]:
        p.add_argument(f"--{flag}", action="store_true")

    # LoRA variant params
    p.add_argument("--gradient-est-n-steps", type=int, default=1)
    p.add_argument("--gora-n-steps", type=int, default=32)
    p.add_argument("--gora-init-method", default="compress")
    p.add_argument("--gora-max-rank", type=int, default=32)
    p.add_argument("--gora-min-rank", type=int, default=4)
    p.add_argument("--relora-steps", type=int, default=None)
    p.add_argument("--lora-plus-scaler", type=int, default=16)
    p.add_argument("--me-lora-n-split", type=int, default=2)
    p.add_argument("--delta-lora-start-steps", type=int, default=None)

    raw = p.parse_args()
    # Convert hyphen-keys to underscore-keys
    d = {k.replace("-", "_"): v for k, v in vars(raw).items()}
    return LoRALLMConfig.from_dict(d)


if __name__ == "__main__":
    cfg = _parse_cli()
    train_lora_llm(cfg)
