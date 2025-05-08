"""Dataset root paths. Override with CROPWEED_ROOT env var."""

from __future__ import annotations

import os


class Path:
    @staticmethod
    def db_root_dir(dataset: str) -> str:
        if dataset == "cropweed":
            return os.environ.get(
                "CROPWEED_ROOT",
                os.path.join(os.path.dirname(__file__), "data", "cropweed"),
            )
        raise ValueError(f"Dataset '{dataset}' is not available.")
