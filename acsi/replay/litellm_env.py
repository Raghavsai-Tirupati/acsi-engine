from __future__ import annotations

import logging
from typing import Any

_CONFIGURED = False


def import_litellm() -> Any:
    """Import litellm with console noise suppressed.

    litellm prints provider-hint banners ("LLM Provider List: ...") and debug
    lines to stdout/stderr at import and on every unmapped-model cost lookup.
    That spam drowned the preflight table and judge progress during the first
    live run. Silencing it is configuration only — routing, cost math, retries,
    and completion behavior are unchanged. Configured once per process.
    """
    import litellm

    global _CONFIGURED
    if not _CONFIGURED:
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        _CONFIGURED = True
    return litellm
