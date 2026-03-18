from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from Chevolet_GraphRAG.config import Settings

logger = logging.getLogger(__name__)


class SafeEmbeddings(Embeddings):
    """Guard embedding calls from overlong text input."""

    def __init__(self, base: Embeddings, max_chars: int) -> None:
        self.base = base
        self.max_chars = max(64, max_chars)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized = [self._normalize_text(t) for t in texts]
        return self.base.embed_documents(normalized)

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self._normalize_text(text))

    def _normalize_text(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", (text or "")).strip()
        if len(clean) <= self.max_chars:
            return clean

        head_len = self.max_chars // 2
        tail_len = self.max_chars - head_len - 5
        return f"{clean[:head_len]} ... {clean[-tail_len:]}"


def _resolve_hf_model_path(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.as_posix()

    if "/" not in model_name:
        return model_name

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name

    ref_file = repo_dir / "refs" / "main"
    if ref_file.exists():
        revision = ref_file.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_dir = snapshots_dir / revision
            if snapshot_dir.exists():
                return snapshot_dir.as_posix()

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if snapshots:
        return snapshots[-1].as_posix()

    return model_name


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)
        except Exception:
            return None

    if provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)
        except Exception:
            return None

    return None


def build_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return SafeEmbeddings(
            base=OpenAIEmbeddings(model=settings.embedding_model),
            max_chars=settings.embedding_text_max_chars,
        )

    model_kwargs: dict[str, Any] = {}
    model_name = settings.embedding_model
    if settings.huggingface_local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_kwargs["local_files_only"] = True
        model_name = _resolve_hf_model_path(model_name)

    from langchain_huggingface import HuggingFaceEmbeddings

    # Multilingual model default for Korean-heavy manuals.
    base = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    return SafeEmbeddings(base=base, max_chars=settings.embedding_text_max_chars)


def build_reranker(settings: Settings):
    """Build a cross-encoder document compressor for reranking."""
    provider = settings.reranker_provider.lower()
    top_n = max(settings.default_top_k * 2, 8)

    if provider == "cohere":
        try:
            from langchain_cohere import CohereRerank

            api_key = settings.cohere_api_key or os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
            if not api_key:
                logger.warning("Cohere reranker disabled: missing COHERE_API_KEY/CO_API_KEY.")
                return None

            os.environ.setdefault("COHERE_API_KEY", api_key)
            os.environ.setdefault("CO_API_KEY", api_key)
            return CohereRerank(
                model=settings.reranker_model,
                top_n=top_n,
                cohere_api_key=api_key,
            )
        except Exception:
            return None

    if provider == "huggingface":
        try:
            model_kwargs: dict[str, Any] = {}
            model_name = settings.reranker_model
            if settings.huggingface_local_files_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                model_kwargs["local_files_only"] = True
                model_name = _resolve_hf_model_path(model_name)

            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            model = HuggingFaceCrossEncoder(
                model_name=model_name,
                model_kwargs=model_kwargs,
            )
            return CrossEncoderReranker(model=model, top_n=top_n)
        except Exception:
            return None
    return None


def invoke_json(chat_model: BaseChatModel | None, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if chat_model is None:
        return fallback

    try:
        response = chat_model.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        return json.loads(content)
    except Exception:
        return fallback
