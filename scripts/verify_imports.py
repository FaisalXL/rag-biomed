#!/usr/bin/env python3
"""Fast check: package imports and syntax (no downloads, no training)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    import adaptive_rag.config  # noqa: F401
    from adaptive_rag import adaptive_shootout  # noqa: F401
    from adaptive_rag import medhallu_data_gen  # noqa: F401
    from adaptive_rag import orchestrate  # noqa: F401
    from adaptive_rag import report_benchmark  # noqa: F401
    from adaptive_rag import train_lora_router  # noqa: F401

    print("verify_imports: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
