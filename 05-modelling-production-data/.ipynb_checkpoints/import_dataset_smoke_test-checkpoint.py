"""Quick smoke test for the ``import_dataset`` helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict


def _load_helper_module():
    """Load helper.py from the current directory via importlib."""
    helper_path = Path(__file__).resolve().parent / "helper.py"
    spec = importlib.util.spec_from_file_location("helper_module", helper_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unable to load helper module from {helper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summarize_default_import(records: Dict[str, Any]) -> None:
    print("=== Default call ===")
    print("states_train:", records["states_train"].shape)
    print("empirical_seq:", records["empirical_seq"].shape)
    print("seq_mask:", records["seq_mask"].shape)
    print("unique_utterances:", records["unique_utterances"].shape)
    print()



def main() -> int:
    try:
        helper = _load_helper_module()
    except Exception as exc:  # noqa: BLE001 - broad for smoke scaffold
        print("Failed to import helper.py:", exc)
        return 1

    try:
        default_records = helper.import_dataset()
        _summarize_default_import(default_records)

        grouped_records = helper.import_empirical_distribution_by_condition()
        _summarize_grouped_import(grouped_records)
    except Exception as exc:  # noqa: BLE001 - surface any runtime issue
        print("import_dataset raised an exception:", exc)
        return 1

    print("Smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
