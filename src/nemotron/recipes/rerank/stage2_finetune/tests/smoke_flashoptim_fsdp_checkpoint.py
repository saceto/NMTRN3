#!/usr/bin/env python3
"""2-GPU FlashAdamW/FSDP2 optimizer checkpoint smoke test.

Run from the repository root:
    PYTHONPATH=src uv run --project src/nemotron/recipes/rerank/stage2_finetune \
        torchrun --standalone --nproc_per_node=2 \
        src/nemotron/recipes/rerank/stage2_finetune/tests/smoke_flashoptim_fsdp_checkpoint.py

This covers the rerank-specific checkpoint issue: a score head shaped
``[num_labels, hidden]`` with ``num_labels=1`` is uneven for default FSDP2
dim-0 sharding, so FlashOptim cannot wrap optimizer state as DTensors for DCP.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from flashoptim import FlashAdamW
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

SRC_ROOT = Path(__file__).resolve().parents[5]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import nemo_automodel.components.distributed.parallelizer as automodel_parallelizer  # noqa: E402
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState  # noqa: E402

from nemotron.recipes.rerank.stage2_finetune.train import (  # noqa: E402
    _patch_flashoptim_fsdp2_shard_placement,
)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _init_dist() -> None:
    if _world_size() != 2:
        raise AssertionError("Run this smoke with exactly 2 ranks: torchrun --nproc_per_node=2 ...")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")


def _cleanup_dist() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _build_sharded_score_head_model() -> nn.Module:
    _patch_flashoptim_fsdp2_shard_placement()
    mesh = init_device_mesh("cuda", (_world_size(),))

    model = nn.Sequential(
        nn.Linear(16, 32, bias=False),
        nn.GELU(),
        nn.Linear(32, 1, bias=False),
    ).cuda()
    model.to(torch.bfloat16)

    for module in model:
        if isinstance(module, nn.Linear):
            automodel_parallelizer.fully_shard(module, mesh=mesh)
    automodel_parallelizer.fully_shard(model, mesh=mesh)
    return model


def _build_optimizer(model: nn.Module) -> FlashAdamW:
    return FlashAdamW(
        model.parameters(),
        lr=1.0e-3,
        quantize=False,
        compress_state_dict=False,
        master_weight_bits=32,
        fused=True,
    )


def _train_one_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    loss = model(x).float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _save_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_dir: Path) -> None:
    state = OptimizerState(model, optimizer).state_dict()
    dcp.save(state, checkpoint_id=str(ckpt_dir))


def _load_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_dir: Path) -> None:
    optimizer_state = OptimizerState(model, optimizer)
    state = optimizer_state.state_dict()
    dcp.load(state, checkpoint_id=str(ckpt_dir))
    optimizer_state.load_state_dict(state)


def _nonempty_optimizer_state_count(optimizer: torch.optim.Optimizer) -> int:
    return sum(1 for state in optimizer.state.values() if state)


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise AssertionError("This smoke requires at least 2 CUDA GPUs.")

    _init_dist()
    try:
        model = _build_sharded_score_head_model()
        optimizer = _build_optimizer(model)
        _train_one_step(model, optimizer)

        ckpt_dir = Path(os.environ.get("CKPT_ROOT", "/tmp/flashoptim-fsdp-ckpt-smoke")) / "optim"
        if _rank() == 0 and ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        dist.barrier()

        _save_optimizer(model, optimizer, ckpt_dir)

        loaded_model = _build_sharded_score_head_model()
        loaded_optimizer = _build_optimizer(loaded_model)
        _load_optimizer(loaded_model, loaded_optimizer, ckpt_dir)

        if _nonempty_optimizer_state_count(loaded_optimizer) == 0:
            raise AssertionError("Loaded FlashAdamW optimizer has no per-parameter state.")

        dist.barrier()
        if _rank() == 0:
            shutil.rmtree(ckpt_dir.parent)

        if _rank() == 0:
            print("FlashAdamW FSDP2 checkpoint save/load smoke: ok")
    finally:
        _cleanup_dist()


if __name__ == "__main__":
    main()
