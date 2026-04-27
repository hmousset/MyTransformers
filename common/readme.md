# Common Package

The `common` package contains shared training infrastructure:

- CLI argument parsing
- named path registration
- distributed and parallel-state setup
- logging and rank-aware printing
- utility functions such as `print_rank_0`, `read_config`, `set_random_seed`, and `init_dist`
- optimizer and scheduler construction
- trainable-parameter management
- LoRA and LoRA-variant implementations
- the registry used by models, tokenizers, datasets, and path aliases

Important: importing `common` loads [paths.py](paths.py), which expects a `paths.json` file in the repository root. Use `{}` if you want to pass every path explicitly on the command line.

## Common Utilities

```python
from common.utils import print_rank_0, read_config, set_random_seed, init_dist
```

- `print_rank_0` prints through the project logger only on rank 0 unless `force_print=True`.
- `read_config` loads JSON or INI config files.
- `set_random_seed` seeds Python, NumPy, and PyTorch.
- `init_dist` initializes DeepSpeed distributed state and the project parallel-state groups.

## Optimizer Setup

```python
import deepspeed

from common.optimizer import get_optimizer

optimizer, lr_scheduler = get_optimizer(ds_config, args, model=model)

model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    config=ds_config,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    mpu=parallel_states,
)
```

Pass the optimizer and scheduler returned by `get_optimizer` into DeepSpeed. Otherwise the custom optimizer settings are not used.

## Trainable Parameter Helpers

```python
from common.utils.params_manager import (
    refresh_config,
    print_trainable_module_names,
    enable_trainable_params,
    disable_untrainable_params,
)
```

- `refresh_config` updates the DeepSpeed config from parsed CLI arguments. CLI arguments have priority.
- `print_trainable_module_names` prints trainable parameter names.
- `enable_trainable_params` enables parameters whose names contain any configured substring.
- `disable_untrainable_params` disables parameters whose names contain any configured substring.

## Parser Usage

Always call the full project parser through `get_args()`:

```python
from common.parser import get_args

args = get_args()
```

Calling the lower-level parser builders manually can leave parts of the argument namespace missing.

## LoRA Usage

```python
from train.load_model import load_model
from common.lora_modules import *

model, tokenizer, model_config, return_dataset_kwargs = load_model(args)

setup_lora(model, args, model_config)

if args.use_lora_ga:
    lora_ga_reinit(
        model=model,
        dataloader=train_dataloader,
        args=args,
        iters=args.gradient_est_n_steps,
    )

if args.use_gora:
    gora_reinit(
        model=model,
        dataloader=train_dataloader,
        args=args,
        iters=args.gradient_est_n_steps,
    )

if args.use_adalora:
    rank_allocator = RankAllocator(model, args)
    model.rankallocator = rank_allocator

if args.use_increlora:
    rank_allocator = IncreRankAllocator(model, args)
    model.rankallocator = rank_allocator
```

The training entry point already performs this setup. Use the snippet only when you build custom training code.
