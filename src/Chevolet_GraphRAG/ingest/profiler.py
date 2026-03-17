from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import fitz

from chevy_troubleshooter.ingest.catalog import discover_manual_files


def profile_dataset(data_root: Path, include_page_counts: bool = False) -> dict:
    catalog = discover_manual_files(data_root)

    model_counter = Counter()
    type_counter = Counter()
    model_type_counter: dict[str, Counter[str]] = defaultdict(Counter)

    total_pages = 0
    page_failures = 0

    for manual in catalog.manuals:
        model_counter[manual.model] += 1
        type_counter[manual.manual_type] += 1
        model_type_counter[manual.model][manual.manual_type] += 1

        if include_page_counts:
            try:
                with fitz.open(manual.file_path) as doc:
                    total_pages += len(doc)
            except Exception:
                page_failures += 1

    profile = {
        "models": len(model_counter),
        "manuals": len(catalog.manuals),
        "manuals_per_model": dict(model_counter),
        "manual_type_distribution": dict(type_counter),
        "model_type_matrix": {k: dict(v) for k, v in model_type_counter.items()},
    }

    if include_page_counts:
        profile["total_pages"] = total_pages
        profile["page_count_failures"] = page_failures

    return profile
