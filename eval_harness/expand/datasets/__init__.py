"""Dataset adapters. Each registers via @register_dataset and implements
``iter_seeds(limit, **kw) -> Iterator[Seed]``.
"""

from . import kernelbench  # noqa: F401
from . import verilog_eval_v2  # noqa: F401
from . import rtllm_v2  # noqa: F401
from . import cvdp_cid003  # noqa: F401
from . import tritonbench_g  # noqa: F401
from . import tritonbench_t  # noqa: F401
from . import archxbench  # noqa: F401
from . import realbench  # noqa: F401
