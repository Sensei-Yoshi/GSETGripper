"""Shared data loaded once for each Streamlit application rerun."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.config import Config, load_config
from modules.datasets import Dataset, DatasetObject, discover_datasets


@dataclass(frozen=True)
class AppContext:
    """Read-only inputs shared by every top-level tab renderer."""

    config: Config
    catalog: tuple[Dataset, ...]
    dataset: Dataset
    rows: list[DatasetObject]
    summary: dict[str, Any]


def load_context(
    dataset_id: str = "expforce",
    *,
    base_config: Config | None = None,
    catalog: list[Dataset] | tuple[Dataset, ...] | None = None,
) -> AppContext:
    """Load one dataset snapshot shared by every tab during this rerun."""
    base_cfg = (base_config or load_config()).model_copy(deep=True)
    available = tuple(catalog or discover_datasets(base_cfg))
    by_id = {dataset.dataset_id: dataset for dataset in available}
    if dataset_id not in by_id:
        if not available:
            raise RuntimeError(f"No dataset folders were found under {base_cfg.root / 'data'}")
        dataset_id = available[0].dataset_id
    dataset = by_id[dataset_id]
    runtime_cfg = dataset.runtime_config(base_cfg)
    return AppContext(
        config=runtime_cfg,
        catalog=available,
        dataset=dataset,
        rows=list(dataset.objects.values()),
        summary=dataset.summary(),
    )
