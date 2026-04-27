# Training Package

The unified training entry point is [u_train.py](u_train.py). Launch it with DeepSpeed or `torchrun` and pass the training arguments defined in [common/parser.py](../common/parser.py).

## Main Flow

1. `common.parser.get_args()` parses CLI arguments, initializes distributed state, validates settings, and fills default DeepSpeed config paths when `--zero-stage` is used.
2. `registry.get_paths(args)` resolves named paths from `paths.json`.
3. `load_model(args)` loads either:
   - a Hugging Face model/tokenizer when `--huggingface` is set, or
   - a local MyTransformers model/tokenizer from the registry.
4. `setup_lora(model, args, model_config)` replaces configured modules with LoRA or LoRA variants when requested.
5. `load_dataloder(...)` builds train and eval dataloaders from the dataset registry.
6. The DeepSpeed config is loaded and refreshed from CLI arguments.
7. Trainable parameters are enabled or disabled.
8. The optimizer and learning-rate scheduler are created.
9. `init_distributed_model(...)` wraps the model with DeepSpeed.
10. `Trainer.train(...)` runs forward, backward, evaluation, logging, and checkpoint saving.

## Important Files

- [u_train.py](u_train.py): main training entry point.
- [load_model.py](load_model.py): Hugging Face and local model loading.
- [load_data.py](load_data.py): dataset selection, dataloader construction, and training-step accounting.
- [dp_train.py](dp_train.py): data-parallel forward, backward, evaluation, and task-print functions.
- [pp_train.py](pp_train.py): pipeline-parallel forward, evaluation, and task-print functions.
- [trainer.py](trainer.py): reusable training loop, logging, evaluation, and checkpoint saving.

## Data Parallel vs Pipeline Parallel

When `--num-pp-stages` is not set, training uses the DeepSpeed data-parallel path in `dp_train.py`.

When `--num-pp-stages` is set, the model is converted to its registered pipeline wrapper and training uses `pp_train.py`. Pipeline checkpoints are saved through DeepSpeed's pipeline checkpoint API rather than as a single `.ckpt` file.

## Notes

- Hugging Face models currently use the data-parallel path. Pipeline and sequence parallelism are only wired for local model wrappers.
- `--train-iters` and `--epochs` are mutually exclusive. If neither is given, the parser defaults to `10000` training iterations.
- If you want checkpoints, set `--output-path` and either `--save-interval` or `--save-epoch`. Without an output path, training runs but checkpoints are not saved.
- If you do not have an eval dataset, pass `--skip-eval`.
- The codebase is registry-driven; see [common/registry.py](../common/registry.py) for how models, tokenizers, datasets, and path names are registered and resolved.
