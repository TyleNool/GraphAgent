from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Chevolet_GraphRAG.models import ManualFile


PDF_SUFFIX = ".pdf"
ZONE_IDENTIFIER = "Zone.Identifier"

MANUAL_TYPE_MAP = {
    "긴급조치": "emergency_action",
    "긴급상황": "emergency_action",
    "응급": "emergency_action",
    "조치방법": "emergency_action",
    "긴급": "emergency_action",
    "개요": "overview",
    "계기판": "cluster_controls",
    "조절장치": "cluster_controls",
    "운전": "driving_operation",
    "작동": "driving_operation",
    "인포테인먼트": "infotainment",
    "MyLink": "infotainment",
    "키": "access_openings",
    "도어": "access_openings",
    "유리창": "access_openings",
    "조명": "lighting",
    "온도조절": "hvac",
    "시트": "seat_safety",
    "안전": "seat_safety",
    "서비스": "service_maintenance",
    "정비": "service_maintenance",
    "차량관리": "vehicle_care",
    "보관": "storage",
    "보증서": "warranty",
    "기술제원": "specs",
}


VEHICLE_CATEGORY_HINTS = {
    "ev": ["BOLT", "VOLT", "EV", "볼트"],
    "sedan": ["말리부", "임팔라", "크루즈", "아베오", "Spark", "카마로", "스파크", "알페온"],
    "suv": ["트레일블레이저", "트래버스", "이쿼녹스", "트랙스", "TAHOE", "타호", "캡티바", "올란도"],
    "truck": ["콜로라도"],
}


@dataclass(slots=True)
class DataCatalog:
    manuals: list[ManualFile]

    @property
    def known_models(self) -> list[str]:
        return sorted({m.model for m in self.manuals})

    @property
    def model_to_category(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for model in self.known_models:
            mapping[model] = infer_vehicle_category(model)
        return mapping

    def summary(self) -> dict[str, object]:
        by_model = Counter(m.model for m in self.manuals)
        by_type = Counter(m.manual_type for m in self.manuals)
        return {
            "manual_count": len(self.manuals),
            "models": len(by_model),
            "manuals_per_model": dict(by_model),
            "manual_type_distribution": dict(by_type),
        }


def infer_vehicle_category(model_name: str) -> str:
    upper = model_name.upper()
    for category, keywords in VEHICLE_CATEGORY_HINTS.items():
        if any(k.upper() in upper for k in keywords):
            return category
    return "unknown"


def normalize_manual_type(raw_name: str) -> str:
    for key, value in MANUAL_TYPE_MAP.items():
        if key in raw_name:
            return value
    return "general"


def _build_model_aliases(data_root: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        aliases[child.name.lower()] = child.name
    return aliases


def _resolve_model_name(raw_model: str, model_aliases: dict[str, str]) -> str:
    if not raw_model:
        return raw_model
    return model_aliases.get(raw_model.lower(), raw_model)


def _infer_model_from_root_stem(stem: str, model_aliases: dict[str, str]) -> str:
    lower_stem = stem.lower()
    candidates = sorted(set(model_aliases.values()), key=len, reverse=True)

    for candidate in candidates:
        c = candidate.lower()
        if lower_stem == c or lower_stem.startswith(f"{c}_"):
            return candidate

    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        return _resolve_model_name(prefix, model_aliases)
    return _resolve_model_name(stem, model_aliases)


def _parse_manual_file(path: Path, data_root: Path, model_aliases: dict[str, str]) -> ManualFile:
    parent_model = path.parent.name
    stem = path.stem

    is_root_level = path.parent.resolve() == data_root.resolve()

    if is_root_level:
        model = _infer_model_from_root_stem(stem, model_aliases)
    else:
        model = _resolve_model_name(parent_model, model_aliases)

    if "_" in stem:
        _, type_hint = stem.split("_", 1)
    else:
        type_hint = stem
    return ManualFile(model=model, manual_type=normalize_manual_type(type_hint), file_path=path)


def discover_manual_files(
    data_root: Path,
    include_models: Iterable[str] | None = None,
    filename_keywords: Iterable[str] | None = None,
) -> DataCatalog:
    include_model_set = {m.strip().lower() for m in (include_models or []) if m.strip()}
    keyword_set = {k.strip().lower() for k in (filename_keywords or []) if k.strip()}

    model_aliases = _build_model_aliases(data_root)
    manuals: list[ManualFile] = []

    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() != PDF_SUFFIX:
            continue
        if ZONE_IDENTIFIER in path.as_posix():
            continue

        manual = _parse_manual_file(path, data_root=data_root, model_aliases=model_aliases)

        if include_model_set and manual.model.lower() not in include_model_set:
            continue

        if keyword_set:
            target = f"{path.name} {path.stem}".lower()
            if not any(keyword in target for keyword in keyword_set):
                continue

        manuals.append(manual)

    return DataCatalog(manuals=manuals)
