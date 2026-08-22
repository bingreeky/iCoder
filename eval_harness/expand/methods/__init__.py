"""Expansion-method adapters. Each registers a class with an async
``expand(seed, llm, num_variants, **kw) -> list[Expanded]``.
"""

from . import xcoder  # noqa: F401
from . import benchevolver  # noqa: F401
from . import inversecoder  # noqa: F401
from . import evol_instruct  # noqa: F401
from . import kb_perturb_be  # noqa: F401
from . import kb_perturb_ei  # noqa: F401
from . import tbg_perturb_be  # noqa: F401
from . import tbg_perturb_ei  # noqa: F401
from . import tbt_perturb_be  # noqa: F401
from . import tbt_perturb_ei  # noqa: F401
