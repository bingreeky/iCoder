"""Unified data-expansion pilot framework.

Layout:
  expand/datasets/   one adapter per source benchmark (KernelBench / VEval v2 /
                    RTLLM v2 / CVDP cid003), each yielding normalised seeds.
  expand/methods/    one adapter per expansion strategy (xcoder, benchevolver,
                    inversecoder). Each consumes a seed and emits N variants.
  expand/llm.py      thin async OpenAI-compatible client + offline dry-run stub.
  expand/registry.py decorator-based dataset / method registry.

Wired up by scripts/expand_data.py and scripts/convert_to_sft.py.
"""

from . import datasets as _datasets  # noqa: F401  (registers adapters)
from . import methods as _methods  # noqa: F401  (registers adapters)
from .registry import DATASETS, METHODS, get_dataset, get_method

__all__ = ["DATASETS", "METHODS", "get_dataset", "get_method"]
