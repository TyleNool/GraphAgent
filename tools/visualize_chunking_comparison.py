#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_REPORT = PROJECT_ROOT / "eval" / "reports" / "Baseline_chunking.json"
DEFAULT_PARENT_REPORT = PROJECT_ROOT / "eval" / "reports" / "parent_child.json"
DEFAULT_BASELINE_IMAGE = PROJECT_ROOT / "eval" / "reports" / "Baseline_Chunking.png"
DEFAULT_PARENT_IMAGE = PROJECT_ROOT / "eval" / "reports" / "parent_child.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "reports" / "chunking_comparison_dashboard.png"


@dataclass(frozen=True)
class MetricSpec:
    section: str
    key: str
    label: str
    kind: str
    higher_is_better: bool


METRICS: list[MetricSpec] = [
    MetricSpec("Routing & Guardrail", "guardrail_accuracy", "Guardrail Accuracy", "percent", True),
    MetricSpec("Routing & Guardrail", "guardrail_f1", "Guardrail F1", "percent", True),
    MetricSpec("Routing & Guardrail", "routing_accuracy", "Routing Accuracy", "percent", True),
    MetricSpec("Routing & Guardrail", "model_family_accuracy", "Model Family Accuracy", "percent", True),
    MetricSpec("Graph & Document Retrieval", "doc_hit_at_5", "Document Hit@5", "percent", True),
    MetricSpec("Graph & Document Retrieval", "doc_mrr", "Document MRR", "percent", True),
    MetricSpec("Graph & Document Retrieval", "entity_hit_rate", "Entity Hit Rate", "percent", True),
    MetricSpec("Graph & Document Retrieval", "page_hit_at_5", "Page Hit@5", "percent", True),
    MetricSpec("Graph & Document Retrieval", "manual_type_match", "Manual Type Match", "percent", True),
    MetricSpec("Generation & Grounding", "groundedness_pass_rate", "Groundedness Pass Rate", "percent", True),
    MetricSpec("Generation & Grounding", "hallucination_rate", "Hallucination Rate", "percent", False),
    MetricSpec("Generation & Grounding", "fact_coverage", "Fact Coverage", "percent", True),
    MetricSpec("Multimodal & UX Alignment", "image_source_alignment", "Image-Source Alignment", "percent", True),
    MetricSpec("Multimodal & UX Alignment", "confidence_bucket_accuracy", "Confidence Bucket Accuracy", "percent", True),
    MetricSpec("Operational Metrics", "latency_p50_sec", "Latency P50", "seconds", False),
    MetricSpec("Operational Metrics", "latency_p95_sec", "Latency P95", "seconds", False),
    MetricSpec("Operational Metrics", "avg_cost_usd", "Average Cost", "currency", False),
    MetricSpec("Operational Metrics", "error_rate", "Error Rate", "percent", False),
    MetricSpec("Operational Metrics", "requery_rate", "Requery Rate", "percent", False),
]


SECTION_ORDER = [
    "Routing & Guardrail",
    "Graph & Document Retrieval",
    "Generation & Grounding",
    "Multimodal & UX Alignment",
    "Operational Metrics",
]

SECTION_TO_SCORECARD_KEY = {
    "Routing & Guardrail": "1_routing_guardrail",
    "Graph & Document Retrieval": "2_graph_document_retrieval",
    "Generation & Grounding": "3_generation_grounding",
    "Multimodal & UX Alignment": "4_multimodal_ux",
    "Operational Metrics": "5_operational",
}


PALETTE = {
    "bg": "#f4efe6",
    "panel": "#fffaf2",
    "panel_alt": "#f9f3e7",
    "ink": "#1d1d1b",
    "muted": "#70695f",
    "grid": "#d7cdbc",
    "baseline": "#d97745",
    "baseline_soft": "#efc2a9",
    "parent": "#177f73",
    "parent_soft": "#9dd5cf",
    "accent": "#214b78",
    "good": "#177f73",
    "bad": "#b4542a",
    "neutral": "#8b8377",
}


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if mono:
        candidates.extend(["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"])
    elif bold:
        candidates.extend(["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"])
    else:
        candidates.extend(["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_metric(scorecard: dict[str, Any], spec: MetricSpec) -> float:
    for section_name, section_key in SECTION_TO_SCORECARD_KEY.items():
        if section_name == spec.section:
            return float(scorecard.get(section_key, {}).get(spec.key, 0.0))
    return 0.0


def _format_value(value: float, kind: str) -> str:
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "seconds":
        return f"{value:.3f}s"
    if kind == "currency":
        return f"${value:.5f}"
    return f"{value:.4f}"


def _format_delta(value: float, kind: str) -> str:
    sign = "+" if value > 0 else ""
    if kind == "percent":
        return f"{sign}{value * 100:.1f} pts"
    if kind == "seconds":
        return f"{sign}{value:.3f}s"
    if kind == "currency":
        return f"{sign}${value:.5f}"
    return f"{sign}{value:.4f}"


def _metric_improvement(spec: MetricSpec, baseline: float, parent: float) -> float:
    return parent - baseline if spec.higher_is_better else baseline - parent


def _winner_label(spec: MetricSpec, baseline: float, parent: float, base_label: str, parent_label: str) -> str:
    improvement = _metric_improvement(spec, baseline, parent)
    if abs(improvement) < 1e-12:
        return "Tie"
    return parent_label if improvement > 0 else base_label


def _normalized_metric_score(spec: MetricSpec, baseline: float, parent: float, target: float) -> float:
    if spec.kind == "percent":
        return target if spec.higher_is_better else 1.0 - target
    span_max = max(baseline, parent, 1e-9)
    score = 1.0 - (target / span_max)
    return max(0.0, min(1.0, score))


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _draw_round_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 0)


def _make_thumbnail(path: Path, size: tuple[int, int]) -> Image.Image | None:
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    thumb = Image.new("RGB", size, PALETTE["panel_alt"])
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    thumb.paste(image, (x, y))
    return thumb


def _draw_summary_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    value: str,
    subtitle: str,
    title_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    fill: str,
) -> None:
    _draw_round_rect(draw, box, fill=fill, outline=PALETTE["grid"], radius=20)
    x1, y1, _, _ = box
    _draw_text(draw, (x1 + 24, y1 + 18), title.upper(), title_font, PALETTE["muted"])
    _draw_text(draw, (x1 + 24, y1 + 48), value, value_font, PALETTE["ink"])
    _draw_text(draw, (x1 + 24, y1 + 112), subtitle, body_font, PALETTE["muted"])


def _draw_metric_row(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    spec: MetricSpec,
    baseline: float,
    parent: float,
    base_label: str,
    parent_label: str,
    label_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    mono_font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    row_h = y2 - y1
    label_x = x1 + 24
    bar_x = x1 + 330
    bar_w = 500
    value_x = bar_x + bar_w + 28
    badge_x = x2 - 240

    direction = "Higher is better" if spec.higher_is_better else "Lower is better"
    _draw_text(draw, (label_x, y1 + 8), spec.label, label_font, PALETTE["ink"])
    _draw_text(draw, (label_x, y1 + 30), direction, body_font, PALETTE["muted"])

    scale_max = 1.0 if spec.kind == "percent" else max(baseline, parent, 1e-9)
    base_ratio = 0.0 if scale_max <= 0 else baseline / scale_max
    parent_ratio = 0.0 if scale_max <= 0 else parent / scale_max

    track_top = y1 + 12
    track_h = 12
    gap = 16
    draw.rounded_rectangle((bar_x, track_top, bar_x + bar_w, track_top + track_h), radius=6, fill="#e7ddcf")
    draw.rounded_rectangle((bar_x, track_top + track_h + gap, bar_x + bar_w, track_top + track_h * 2 + gap), radius=6, fill="#e7ddcf")
    draw.rounded_rectangle(
        (bar_x, track_top, int(bar_x + bar_w * base_ratio), track_top + track_h),
        radius=6,
        fill=PALETTE["baseline"],
    )
    draw.rounded_rectangle(
        (bar_x, track_top + track_h + gap, int(bar_x + bar_w * parent_ratio), track_top + track_h * 2 + gap),
        radius=6,
        fill=PALETTE["parent"],
    )

    _draw_text(draw, (value_x, y1 + 6), f"{base_label}: {_format_value(baseline, spec.kind)}", mono_font, PALETTE["baseline"])
    _draw_text(draw, (value_x, y1 + 30), f"{parent_label}: {_format_value(parent, spec.kind)}", mono_font, PALETTE["parent"])

    improvement = _metric_improvement(spec, baseline, parent)
    winner = _winner_label(spec, baseline, parent, base_label, parent_label)
    badge_fill = PALETTE["good"] if winner == parent_label else PALETTE["bad"]
    if winner == "Tie":
        badge_fill = PALETTE["neutral"]
    badge = f"{winner}  {_format_delta(improvement, spec.kind)}"
    badge_box = (badge_x, y1 + 10, x2 - 24, y1 + row_h - 10)
    _draw_round_rect(draw, badge_box, fill=badge_fill, radius=16)
    text_w, text_h = _text_size(draw, badge, body_font)
    text_x = badge_box[0] + max(16, ((badge_box[2] - badge_box[0]) - text_w) // 2)
    text_y = badge_box[1] + ((badge_box[3] - badge_box[1]) - text_h) // 2 - 1
    _draw_text(draw, (text_x, text_y), badge, body_font, "#ffffff")


def create_dashboard(
    baseline_report: Path,
    parent_report: Path,
    output_path: Path,
    baseline_label: str,
    parent_label: str,
    baseline_image: Path | None = None,
    parent_image: Path | None = None,
) -> Path:
    baseline = _read_report(baseline_report)
    parent = _read_report(parent_report)
    baseline_scorecard = baseline.get("scorecard", {})
    parent_scorecard = parent.get("scorecard", {})

    metric_rows = [
        {
            "spec": spec,
            "baseline": _safe_metric(baseline_scorecard, spec),
            "parent": _safe_metric(parent_scorecard, spec),
        }
        for spec in METRICS
    ]

    parent_wins = sum(
        1
        for row in metric_rows
        if _metric_improvement(row["spec"], row["baseline"], row["parent"]) > 1e-12
    )
    baseline_wins = sum(
        1
        for row in metric_rows
        if _metric_improvement(row["spec"], row["baseline"], row["parent"]) < -1e-12
    )
    ties = len(metric_rows) - parent_wins - baseline_wins

    strongest_gain = max(metric_rows, key=lambda row: _metric_improvement(row["spec"], row["baseline"], row["parent"]))
    largest_drop = min(metric_rows, key=lambda row: _metric_improvement(row["spec"], row["baseline"], row["parent"]))

    base_norm = sum(
        _normalized_metric_score(row["spec"], row["baseline"], row["parent"], row["baseline"])
        for row in metric_rows
    ) / len(metric_rows)
    parent_norm = sum(
        _normalized_metric_score(row["spec"], row["baseline"], row["parent"], row["parent"])
        for row in metric_rows
    ) / len(metric_rows)

    section_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SECTION_ORDER}
    for row in metric_rows:
        section_rows[row["spec"].section].append(row)

    title_font = _load_font(44, bold=True)
    section_font = _load_font(26, bold=True)
    title_small_font = _load_font(15, bold=True)
    value_font = _load_font(40, bold=True)
    label_font = _load_font(19, bold=True)
    body_font = _load_font(16)
    mono_font = _load_font(17, mono=True)

    width = 1700
    top_h = 410
    section_title_h = 56
    row_h = 62
    footer_h = 72
    card_gap = 22
    outer_pad = 36
    section_heights = [section_title_h + row_h * len(section_rows[name]) + 28 for name in SECTION_ORDER]
    height = outer_pad + top_h + 24 + sum(section_heights) + card_gap * (len(SECTION_ORDER) - 1) + footer_h + outer_pad

    canvas = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(canvas)

    _draw_text(draw, (outer_pad, 26), "Chunking Evaluation Comparison", title_font, PALETTE["ink"])
    _draw_text(
        draw,
        (outer_pad, 78),
        "Parent/Child vs Baseline across the same 300-item GraphRAG evaluation set",
        label_font,
        PALETTE["muted"],
    )

    summary_box = (outer_pad, 118, 980, 118 + top_h - 30)
    _draw_round_rect(draw, summary_box, fill=PALETTE["panel"], outline=PALETTE["grid"], radius=26)
    _draw_text(draw, (summary_box[0] + 28, summary_box[1] + 24), "At a glance", section_font, PALETTE["accent"])

    card_y = summary_box[1] + 84
    _draw_summary_card(
        draw,
        (summary_box[0] + 28, card_y, summary_box[0] + 280, card_y + 150),
        "Parent/Child Wins",
        f"{parent_wins} / {len(metric_rows)}",
        f"{parent_label} leads in {parent_wins} metrics.",
        title_small_font,
        value_font,
        body_font,
        PALETTE["panel_alt"],
    )
    _draw_summary_card(
        draw,
        (summary_box[0] + 302, card_y, summary_box[0] + 554, card_y + 150),
        "Baseline Wins",
        f"{baseline_wins} / {len(metric_rows)}",
        f"{baseline_label} leads in {baseline_wins} metrics.",
        title_small_font,
        value_font,
        body_font,
        PALETTE["panel_alt"],
    )
    _draw_summary_card(
        draw,
        (summary_box[0] + 576, card_y, summary_box[0] + 828, card_y + 150),
        "Normalized Score",
        f"{parent_norm * 100:.1f} vs {base_norm * 100:.1f}",
        "Higher means stronger aggregate performance after direction-aware normalization.",
        title_small_font,
        _load_font(28, bold=True),
        body_font,
        PALETTE["panel_alt"],
    )

    gain = _metric_improvement(strongest_gain["spec"], strongest_gain["baseline"], strongest_gain["parent"])
    drop = _metric_improvement(largest_drop["spec"], largest_drop["baseline"], largest_drop["parent"])
    _draw_text(draw, (summary_box[0] + 28, summary_box[1] + 262), "Key movement", section_font, PALETTE["accent"])
    _draw_text(
        draw,
        (summary_box[0] + 30, summary_box[1] + 310),
        f"Biggest gain: {strongest_gain['spec'].label} ({_format_delta(gain, strongest_gain['spec'].kind)})",
        label_font,
        PALETTE["good"],
    )
    _draw_text(
        draw,
        (summary_box[0] + 30, summary_box[1] + 338),
        f"Largest drop: {largest_drop['spec'].label} ({_format_delta(drop, largest_drop['spec'].kind)})",
        label_font,
        PALETTE["bad"],
    )
    _draw_text(
        draw,
        (summary_box[0] + 30, summary_box[1] + 366),
        f"Ties: {ties} metrics. Source values are taken from the report JSONs that produced the two scorecard PNGs.",
        body_font,
        PALETTE["muted"],
    )

    thumb_w, thumb_h = 310, 250
    thumb_gap = 22
    right_x = 1020
    for idx, (label, path, color) in enumerate(
        [
            (baseline_label, baseline_image, PALETTE["baseline"]),
            (parent_label, parent_image, PALETTE["parent"]),
        ]
    ):
        box_x = right_x + idx * (thumb_w + thumb_gap)
        box = (box_x, 118, box_x + thumb_w, 118 + top_h - 30)
        _draw_round_rect(draw, box, fill=PALETTE["panel"], outline=PALETTE["grid"], radius=26)
        _draw_text(draw, (box_x + 22, 142), label, section_font, color)
        _draw_text(draw, (box_x + 22, 178), "Original scorecard PNG", body_font, PALETTE["muted"])
        thumb = _make_thumbnail(path, (thumb_w - 44, thumb_h))
        if thumb is not None:
            canvas.paste(thumb, (box_x + 22, 208))

    y = outer_pad + top_h + 120
    for section_name in SECTION_ORDER:
        rows = section_rows[section_name]
        section_h = section_title_h + row_h * len(rows) + 28
        box = (outer_pad, y, width - outer_pad, y + section_h)
        _draw_round_rect(draw, box, fill=PALETTE["panel"], outline=PALETTE["grid"], radius=26)
        _draw_text(draw, (box[0] + 24, box[1] + 18), section_name, section_font, PALETTE["accent"])
        _draw_text(
            draw,
            (box[2] - 250, box[1] + 22),
            "Direction-aware comparison",
            body_font,
            PALETTE["muted"],
        )
        row_y = box[1] + section_title_h
        for idx, row in enumerate(rows):
            if idx > 0:
                draw.line((box[0] + 18, row_y, box[2] - 18, row_y), fill=PALETTE["grid"], width=1)
            _draw_metric_row(
                draw,
                (box[0], row_y, box[2], row_y + row_h),
                row["spec"],
                row["baseline"],
                row["parent"],
                baseline_label,
                parent_label,
                label_font,
                body_font,
                mono_font,
            )
            row_y += row_h
        y += section_h + card_gap

    footer = (
        f"Baseline report: {baseline_report.name} | Parent/Child report: {parent_report.name} | "
        f"Output: {output_path.name}"
    )
    _draw_text(draw, (outer_pad, height - footer_h), footer, body_font, PALETTE["muted"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a chunking comparison dashboard from GraphRAG reports")
    parser.add_argument("--baseline-report", default=str(DEFAULT_BASELINE_REPORT))
    parser.add_argument("--parent-report", default=str(DEFAULT_PARENT_REPORT))
    parser.add_argument("--baseline-image", default=str(DEFAULT_BASELINE_IMAGE))
    parser.add_argument("--parent-image", default=str(DEFAULT_PARENT_IMAGE))
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--parent-label", default="Parent/Child")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output = create_dashboard(
        baseline_report=Path(args.baseline_report).resolve(),
        parent_report=Path(args.parent_report).resolve(),
        output_path=Path(args.output).resolve(),
        baseline_label=args.baseline_label,
        parent_label=args.parent_label,
        baseline_image=Path(args.baseline_image).resolve() if args.baseline_image else None,
        parent_image=Path(args.parent_image).resolve() if args.parent_image else None,
    )
    print(f"Visualization saved: {output}")


if __name__ == "__main__":
    main()
