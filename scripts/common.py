#!/usr/bin/env python3
"""Shared runtime helpers for pneumo_prokBERT scripts (no upstream code here)."""
import torch


def add_device_args(p):
    g = p.add_argument_group("device")
    g.add_argument("--gpu", type=int, default=None,
                   help="CUDA device index to use (default: auto = GPU with most free memory)")
    g.add_argument("--cpu", action="store_true",
                   help="force CPU even if CUDA is available")
    return p


def resolve_device(gpu=None, cpu=False):
    """Silent auto-detect: GPU with most free memory if CUDA is available, else CPU.

    No flags needed in the common case. Returns (device_str, is_cuda).
    WS GPU map (CUDA order != nvidia-smi order): CUDA 0 = 5090, 1 = 3090, 2 = 2080Ti.
    Always use CUDA_VISIBLE_DEVICES=0 for the 5090.
    """
    if cpu or not torch.cuda.is_available():
        return "cpu", False
    n = torch.cuda.device_count()
    if gpu is not None:
        if gpu < 0 or gpu >= n:
            raise SystemExit(f"--gpu {gpu} out of range (0..{n - 1})")
        return f"cuda:{gpu}", True
    best, best_free = 0, -1
    for i in range(n):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}", True
