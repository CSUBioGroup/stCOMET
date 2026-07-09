#!/usr/bin/env python
import os
import random

os.environ.setdefault("PYTHONHASHSEED", "41")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

_DEFAULT_STCOMET_SEED = 41


def _initialize_stcomet_seed(seed=_DEFAULT_STCOMET_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)


_initialize_stcomet_seed()

__author__ = "stCOMET"
__email__ = ""

from .stCOMET import stCOMET
from .stcomet_utils import stcomet_spatial_clustering
from .stcomet_preprocess import (
    preprocess_stcomet_adjacency,
    preprocess_stcomet,
    construct_stcomet_spatial_graph,
    get_stcomet_features,
    permute_stcomet_features,
    fix_stcomet_seed,
)

__all__ = [
    "stCOMET",
    "stcomet_spatial_clustering",
    "preprocess_stcomet_adjacency",
    "preprocess_stcomet",
    "construct_stcomet_spatial_graph",
    "get_stcomet_features",
    "permute_stcomet_features",
    "fix_stcomet_seed",
]
