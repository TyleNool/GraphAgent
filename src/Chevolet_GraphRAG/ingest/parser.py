from __future__ import annotations

import contextlib
import io
import logging
import re
import warnings
from pathlib import Path

import fitz

from Chevolet_GraphRAG.models import (
    ChunkArtifact,
    ManualFile,
    PageArtifact,
    ParsedManual,
    build_manual_artifact_slug,
)


class PdfManualParser:
    def __init__(
        self,
        artifact_root: Path,
        chunk_size: int = 420,
        chunk_overlap: int = 80,
        use_docling: bool = False,
    ) -> None:
        self.artifact_root = artifact_root
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_docling = use_docling
        self._configure_ocr_logging()

    def parse(self, manual: ManualFile) -> ParsedManual:
        docling_chunks = self._extract_docling_chunks(manual.file_path) if self.use_docling else {}

        pages: list[PageArtifact] = []
        with fitz.open(manual.file_path) as doc:
            for idx, page in enumerate(doc, start=1):
                has_three_cols = self._detect_three_column_layout(page)
                blocks = self._extract_text_blocks(page, has_three_cols)

                if sum(len(block) for block in blocks) < 60:
                    fallback_text = self._ocr_fallback(page)
                    fallback_blocks = self._fallback_blocks_from_text(fallback_text)
                    if idx in docling_chunks and sum(len(chunk) for chunk in docling_chunks[idx]) > len(fallback_text):
                        blocks = self._fallback_blocks_from_text("\n\n".join(docling_chunks[idx]))
                    else:
                        blocks = fallback_blocks
                elif idx in docling_chunks and sum(len(block) for block in blocks) < 160:
                    blocks.extend(
                        chunk for chunk in docling_chunks[idx]
                        if chunk.strip() and chunk.strip() not in blocks
                    )

                chunks = self._build_chunks_from_blocks(blocks)
                text = "\n\n".join(chunk.text for chunk in chunks).strip()
                page_image = self._render_page_image(manual, page, idx)

                pages.append(
                    PageArtifact(
                        page_no=idx,
                        display_page_label=self._extract_display_page_label(blocks),
                        text=text,
                        image_path=page_image,
                        chunks=chunks,
                        has_three_column_layout=has_three_cols,
                    )
                )

        return ParsedManual(manual=manual, pages=pages)

    def _extract_docling_chunks(self, pdf_path: Path) -> dict[int, list[str]]:
        try:
            from langchain_docling import DoclingLoader

            loader = DoclingLoader(file_path=pdf_path.as_posix())
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                docs = loader.load()

            page_map: dict[int, list[str]] = {}
            for doc in docs:
                content = (doc.page_content or "").strip()
                if len(content) < 5:
                    continue
                page_no = int(doc.metadata.get("page", 0) or 0)
                page_map.setdefault(page_no, []).append(content)
            return page_map
        except Exception:
            return {}

    def _extract_text_blocks(self, page: fitz.Page, has_three_cols: bool) -> list[str]:
        blocks = page.get_text("blocks")
        if not blocks:
            return []

        page_width = page.rect.width
        normalized: list[tuple[tuple[float, float, float], str]] = []
        for raw in blocks:
            x0, y0, x1, _y1, text, *_ = raw
            clean = self._normalize_block_text(text)
            if len(clean) < 3:
                continue
            normalized.append((self._block_sort_key(x0, x1, y0, page_width, has_three_cols), clean))

        normalized.sort(key=lambda item: item[0])
        return [text for _, text in normalized]

    def _block_sort_key(
        self,
        x0: float,
        x1: float,
        y0: float,
        page_width: float,
        has_three_cols: bool,
    ) -> tuple[float, float, float]:
        center = (x0 + x1) / 2
        if not has_three_cols:
            return (y0, x0, center)

        third = page_width / 3
        if center < third:
            col = 0
        elif center < third * 2:
            col = 1
        else:
            col = 2
        return (col, y0, x0)

    def _detect_three_column_layout(self, page: fitz.Page) -> bool:
        blocks = page.get_text("blocks")
        if not blocks:
            return False

        centers: list[float] = []
        page_width = page.rect.width
        for x0, y0, x1, y1, text, *_ in blocks:
            if not text or not text.strip():
                continue
            centers.append((x0 + x1) / 2)

        if len(centers) < 8:
            return False

        left = sum(1 for c in centers if c < page_width / 3)
        middle = sum(1 for c in centers if page_width / 3 <= c < 2 * page_width / 3)
        right = sum(1 for c in centers if c >= 2 * page_width / 3)

        active_columns = sum(1 for count in (left, middle, right) if count >= 2)
        return active_columns >= 3

    def _ocr_fallback(self, page: fitz.Page) -> str:
        try:
            import pytesseract
            from PIL import Image

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image, lang="kor+eng")
            if len(text.strip()) >= 20:
                return text

            # If whole-page OCR fails, retry by 3-column crops.
            width, height = image.size
            column_texts: list[str] = []
            for idx in range(3):
                left = int(idx * width / 3)
                right = int((idx + 1) * width / 3)
                cropped = image.crop((left, 0, right, height))
                chunk = pytesseract.image_to_string(cropped, lang="kor+eng").strip()
                if chunk:
                    column_texts.append(chunk)

            merged = "\n\n".join(column_texts).strip()
            if merged:
                return merged
            return page.get_text("text")
        except Exception:
            return page.get_text("text")

    def _fallback_blocks_from_text(self, text: str) -> list[str]:
        rows = [self._normalize_block_text(part) for part in re.split(r"\n{2,}", text or "")]
        rows = [row for row in rows if row]
        if rows:
            return rows
        single = self._normalize_block_text(text or "")
        return [single] if single else []

    def _build_chunks_from_blocks(self, blocks: list[str]) -> list[ChunkArtifact]:
        if not blocks:
            return []

        min_chars = max(90, self.chunk_size // 3)
        max_chars = max(min_chars + 40, self.chunk_size)

        units: list[str] = []
        for block in blocks:
            units.extend(self._split_long_block(block, max_chars=max_chars))

        chunks: list[ChunkArtifact] = []
        buffer: list[str] = []
        for unit in units:
            candidate = "\n\n".join(buffer + [unit]).strip()
            if buffer and len(candidate) > max_chars and len("\n\n".join(buffer).strip()) >= min_chars:
                chunks.append(
                    ChunkArtifact(
                        chunk_order=len(chunks) + 1,
                        text="\n\n".join(buffer).strip(),
                    )
                )
                buffer = [unit]
                continue

            buffer.append(unit)
            buffered = "\n\n".join(buffer).strip()
            if len(buffered) >= min_chars and len(buffer) == 1:
                chunks.append(
                    ChunkArtifact(
                        chunk_order=len(chunks) + 1,
                        text=buffered,
                    )
                )
                buffer = []

        if buffer:
            remainder = "\n\n".join(buffer).strip()
            if chunks and len(remainder) < max(50, min_chars // 2):
                chunks[-1].text = f"{chunks[-1].text}\n\n{remainder}".strip()
            else:
                chunks.append(
                    ChunkArtifact(
                        chunk_order=len(chunks) + 1,
                        text=remainder,
                    )
                )

        return [chunk for chunk in chunks if chunk.text.strip()]

    def _split_long_block(self, text: str, max_chars: int) -> list[str]:
        clean = self._normalize_block_text(text)
        if len(clean) <= max_chars:
            return [clean] if clean else []

        sentences = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", clean)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if len(sentences) <= 1:
            return [clean[idx: idx + max_chars].strip() for idx in range(0, len(clean), max_chars)]

        parts: list[str] = []
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if buffer and len(candidate) > max_chars:
                parts.append(buffer.strip())
                buffer = sentence
            else:
                buffer = candidate
        if buffer:
            parts.append(buffer.strip())
        return parts

    def _normalize_block_text(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", (text or "")).strip()
        return clean

    def _extract_display_page_label(self, blocks: list[str]) -> str | None:
        edge_candidates = blocks[:2] + blocks[-3:]
        patterns = [
            re.compile(r"^(\d{1,4})$"),
            re.compile(r"^[Pp]age\s*(\d{1,4})$"),
            re.compile(r"^(\d{1,4})\s*/\s*\d{1,4}$"),
            re.compile(r"^[^\d]{0,20}\s(\d{1,4})$"),
        ]
        for candidate in reversed(edge_candidates):
            text = candidate.strip()
            if len(text) > 16:
                continue
            for pattern in patterns:
                match = pattern.match(text)
                if match:
                    return match.group(1)
        return None

    def _configure_ocr_logging(self) -> None:
        warnings.filterwarnings(
            "ignore",
            message=(
                "Token indices sequence length is longer than the specified "
                "maximum sequence length for this model"
            ),
        )
        logging.getLogger("rapidocr_onnxruntime").setLevel(logging.ERROR)
        logging.getLogger("docling").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        try:
            from transformers import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:
            return

    def _render_page_image(self, manual: ManualFile, page: fitz.Page, page_no: int) -> Path:
        out_dir = (
            self.artifact_root
            / "pages"
            / manual.model
            / build_manual_artifact_slug(manual)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"page_{page_no:04d}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
        pix.save(out_path.as_posix())
        return out_path
