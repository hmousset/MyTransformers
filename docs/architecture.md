# Architecture Guide

This guide is the English Markdown replacement for the original Chinese usage PDF. It explains the main training path and the extension points used by the project.

## Training Entry Point

The main training file is [train/u_train.py](../train/u_train.py).

At startup it performs these steps:

1. Import model, tokenizer, dataset, and LoRA modules so their registry decorators run.
2. Parse arguments with `common.parser.get_args()`.
3. Resolve local paths with `registry.get_paths(args)`.
4. Load the model and tokenizer with `train.load_model.load_model(args)`.
5. Replace target layers with LoRA modules when LoRA or a LoRA variant is enabled.
6. Build train and eval dataloaders.
7. Load and refresh the DeepSpeed config.
8. Run any LoRA preparation step that needs data, such as GoRA or LoRA-GA initialization.
9. Enable and disable trainable parameters.
10. Build the optimizer and learning-rate scheduler.
11. Wrap the model with DeepSpeed.
12. Select the data-parallel or pipeline-parallel forward/backward functions.
13. Run `Trainer.train(...)`.

## Argument Parsing

[common/parser.py](../common/parser.py) builds the complete argument namespace from several parser groups:

- base paths and logging options
- training hyperparameters
- dataset options
- LoRA and parameter-efficient fine-tuning options
- multimodal options
- optimizer options
- DeepSpeed and parallelism options

Always use:

```python
from common.parser import get_args

args = get_args()
```

The parser also initializes distributed state, validates conflicting options, selects the default dtype from `--fp16` or `--bf16`, sets a default `--train-iters 10000` when no duration is provided, and fills `--ds-config-path` from `--zero-stage`.

## Registry System

[common/registry.py](../common/registry.py) holds mappings for:

- model classes
- train-model wrappers
- pipeline-model wrappers
- model config factories
- tokenizer classes
- dataset classes
- named filesystem paths

Local models and datasets register themselves with decorators such as:

```python
@registry.register_model("llama3")
class Llama3(...):
    ...

@registry.register_dataset("iterable")
class BaseIterableDataset(...):
    ...
```

This lets the training path stay generic. For example, `load_model(args)` can ask the registry for the class matching `--model-name` and `--variant` instead of hard-coding one model.

## Path Resolution

[common/paths.py](../common/paths.py) reads `paths.json` from the repository root and registers each nested entry as a path alias.

Given:

```json
{
  "tokenizer": {"llama3": "/models/llama3/tokenizer.model"},
  "model": {"llama3_8b": "/models/llama3/model.ckpt"},
  "train_dataset": {"sft": "/data/train.jsonl"}
}
```

The registered keys are:

- `tokenizer_llama3`
- `model_llama3_8b`
- `train_dataset_sft`

`registry.get_paths(args)` fills missing path arguments from these aliases.

## Model Loading

[train/load_model.py](../train/load_model.py) supports two paths.

### Hugging Face Models

Set `--huggingface` and pass `--model-name-or-path`, or configure a `huggingface_<model>_<variant>` entry in `paths.json`.

The loader:

- calls `AutoModelForCausalLM.from_pretrained(...)`
- tries `attn_implementation="sdpa"` first, then falls back to `"eager"`
- loads the tokenizer with `AutoTokenizer`
- patches the forward method so DeepSpeed training receives `(loss, metric_dict)`
- optionally enables gradient checkpointing
- optionally loads a partial checkpoint for adapter weights

Hugging Face models currently use data-parallel training in this codebase.

### Local MyTransformers Models

Without `--huggingface`, the loader:

- creates a tokenizer from the tokenizer registry
- builds a model config from `<model_name>_<variant>`
- creates the model class from the model registry
- optionally loads a pretrained checkpoint with `--from-pretrained`
- attaches multimodal encoders when requested
- fills model shape arguments used later by training code
- converts the model into either a data-parallel train wrapper or a pipeline wrapper
- moves the model to the requested dtype and device

## LoRA Setup

LoRA setup lives under [common/lora_modules](../common/lora_modules).

The normal training path calls:

```python
setup_lora(model, args, model_config)
```

When `--use-lora` is set, target modules are replaced by LoRA-aware layers. If `--replace-modules` is not provided, local models use `model_config.lora_layers`. Hugging Face models usually need explicit target names such as:

```bash
--replace-modules q_proj k_proj v_proj o_proj
```

Some LoRA variants need data-dependent initialization or a rank allocator:

- `--use-gora` runs GoRA initialization from training batches.
- `--use-lora-ga` estimates gradients before normal training.
- `--use-adalora` attaches a `RankAllocator`.
- `--use-increlora` attaches an incremental rank allocator.

The training entry point handles these setup calls for normal runs.

## Dataset Loading

[train/load_data.py](../train/load_data.py) selects a dataset class from the registry and wraps it in a PyTorch `DataLoader`.

Supported dataset names include:

- `normal`
- `iterable`
- `concat`
- `concat_iterable`
- `multimodal_dna_dataset`
- `iterable_multimodal_dna_dataset`

The common dataset config includes:

- `max_len`
- `max_src_len`
- `mode`
- `meta_prompt`
- `prefix`
- `postfix`
- `input_field`
- `output_field`
- padding or packing behavior
- optional Hugging Face chat-template formatting

In `sft` mode, samples must provide output text. In `pretrain` mode, a line may be plain text, JSON with an input field, or JSON with input and output fields.

## DeepSpeed Setup

DeepSpeed config files are in [ds_config](../ds_config):

- `zero2_config.json`
- `zero3_config.json`
- `pp_config.json`

You can pass one explicitly:

```bash
--ds-config-path ds_config/zero2_config.json
```

Or use:

```bash
--zero-stage 2
```

The parser fills the matching config path, and `refresh_config(ds_config, args)` updates batch size, precision, optimizer, logging, and ZeRO-related fields from CLI arguments.

## Trainer

[train/trainer.py](../train/trainer.py) owns the training loop. It receives:

- a DeepSpeed engine
- train and eval dataloaders
- a forward step
- a backward step
- an eval step
- an optimizer and scheduler
- an optional profiler

The loop performs:

1. forward pass
2. loss and metric accumulation
3. backward and optimizer step
4. profiler step if enabled
5. periodic evaluation
6. rank-0 metric printing
7. checkpoint saving
8. final checkpoint saving when configured

For data-parallel models, checkpoints are normal `.ckpt` files. For pipeline-parallel models, checkpoints are saved as DeepSpeed pipeline checkpoint directories.

## Adding a Local Model

A local model package should usually provide four files.

### `config.py`

Define a dataclass of model hyperparameters and register one or more config factories:

```python
@registry.register_model_config("my_model_7b")
def get_config_for_7b():
    return MyModelConfig(...)
```

### `model.py`

Define the actual PyTorch model and optional generation/inference helpers, then register it:

```python
@registry.register_model("my_model")
class MyModel(...):
    ...
```

### `train_model.py`

Define the data-parallel training wrapper. It should accept only `(model, args)` in `__init__`, inherit from `BaseModel`, and implement:

- `embedding(...)`
- `model_forward(...)`
- optionally `compute_loss(...)` or `compute_metric(...)`

Register it:

```python
@registry.register_train_model("my_model")
class MyModelTrainModel(BaseModel):
    ...
```

### `pipeline_model.py`

Define pipeline layers compatible with DeepSpeed PipelineModule and register a pipeline wrapper:

```python
@registry.register_pipeline_model("my_model")
def get_pipeline_model(model, args):
    return PipelineModule(...)
```

Finally, import the new package in [model/\_\_init\_\_.py](../model/__init__.py) so decorators run at startup.

## Common Utilities

[common/utils/utils.py](../common/utils/utils.py) includes:

- `Timer`
- `print_rank_0`
- `configure_logging`
- `load_ckpt`
- `load_ckpt_for_train`
- `get_merged_state_dict`
- `init_dist`
- `init_distributed_model`
- dataset collators
- metric helpers

[common/optimizer.py](../common/optimizer.py) builds optimizers and scheduler objects. When `--diy-optimizer` is enabled, it can override DeepSpeed's optimizer section and return a custom optimizer and scheduler to `deepspeed.initialize`.

[common/scheduler.py](../common/scheduler.py) implements the project learning-rate schedulers, including cosine and cosine-restart schedules.

