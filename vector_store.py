from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from analysis_work_content import (
    default_chunk_overlap,
    default_chunk_size,
    default_match_threshold,
    EmbeddingError,
    max_query_chars,
    JobChunk,
    aggregate_chunk_scores,
    build_job_chunk_records,
    clean_text,
    job_benefits,
    job_city,
)
from config import get as cfg
from logging_config import get_logger

logger = get_logger(__name__)

vector_schema_version = "chunks-v2"
_index_fields = (
    ("job_id", "keyword"),
    ("city_name", "keyword"),
)


class VectorStoreUnavailable(RuntimeError):
    """Qdrant 未安装、未配置或当前不可用。"""

    error_source = "embedding"


def _enabled() -> bool:
    return cfg("qdrant_enabled", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _model_name() -> str:
    return cfg("embedding_openai_model") or cfg(
        "embedding_model", "text-embedding-3-small"
    )


def _collection_name(model: str) -> str:
    prefix = cfg("qdrant_collection", "jobs").strip() or "jobs"
    provider = cfg("embedding_openai_base_url") or cfg("openai_base_url")
    suffix = hashlib.sha1(f"{provider}\0{model}".encode()).hexdigest()[:10]
    return f"{prefix}_{vector_schema_version}_{suffix}"


def _job_id(card: dict) -> str:
    value = str(
        card.get("encryptJobId")
        or card.get("jobId")
        or ""
    ).strip()
    if value:
        return value
    raw = "|".join([
        clean_text(card.get("jobName")),
        clean_text(card.get("brandName") or card.get("brandComName")),
        str(card.get("lid") or "").strip(),
    ])
    if not raw.strip("|"):
        raw = json.dumps(card, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _content_hash(card: dict) -> str:
    indexed_data = {
        "schema_version": vector_schema_version,
        "chunk_size": default_chunk_size,
        "chunk_overlap": default_chunk_overlap,
        "chunks": [
            {"kind": chunk.kind, "text": chunk.text}
            for chunk in build_job_chunk_records(card)
        ],
        "city_name": job_city(card),
        "city_code": str(card.get("city") or ""),
        "job_experience": str(card.get("jobExperience") or ""),
        "job_degree": str(card.get("jobDegree") or ""),
        "brand_scale_name": str(card.get("brandScaleName") or ""),
        "salary_desc": str(card.get("salaryDesc") or ""),
        "welfare_list": job_benefits(card),
    }
    serialized = json.dumps(indexed_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _chunks(card: dict) -> list[JobChunk]:
    return build_job_chunk_records(card)


def _chunk_point_id(job_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"boss-agent/job/{job_id}/chunk/{chunk_index}",
    ))


def _payload(
    card: dict,
    job_id: str,
    model: str,
    chunk: str,
    chunk_type: str,
    chunk_index: int,
    chunk_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": vector_schema_version,
        "job_id": job_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk": chunk,
        "chunk_type": chunk_type,
        "job_name": str(card.get("jobName") or ""),
        "brand_name": str(card.get("brandName") or ""),
        "city_name": job_city(card),
        "city_code": str(card.get("city") or ""),
        "job_experience": str(card.get("jobExperience") or ""),
        "job_degree": str(card.get("jobDegree") or ""),
        "brand_scale_name": str(card.get("brandScaleName") or ""),
        "salary_desc": str(card.get("salaryDesc") or ""),
        "welfare_list": job_benefits(card),
        "content_hash": _content_hash(card),
        "embedding_model": model,
    }


class QdrantJobStore:
    """职位向量的增量索引和带元数据过滤的召回。"""

    def __init__(self, client=None):
        self._client = client
        self._lock = RLock()
        self._indexed_collections: set[str] = set()
        self.enabled_override: bool | None = None

    @property
    def enabled(self) -> bool:
        if self.enabled_override is not None:
            return self.enabled_override
        return _enabled()

    def _get_client(self):
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise VectorStoreUnavailable(
                    "未安装 qdrant-client，请执行 pip install -r request.txt"
                ) from exc

            url = cfg("qdrant_url").strip()
            api_key = cfg("qdrant_api_key").strip() or None
            timeout = int(cfg("qdrant_timeout", "10") or "10")
            if url:
                self._client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    timeout=timeout,
                )
            else:
                path = Path(cfg("qdrant_path") or Path(__file__).parent / "data" / "qdrant")
                if not path.is_absolute():
                    path = Path(__file__).parent / path
                path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path))
            return self._client

    def _ensure_collection(self, collection: str, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        try:
            exists = client.collection_exists(collection)
        except Exception as exc:
            raise VectorStoreUnavailable(
                f"检查 Qdrant collection 失败: {type(exc).__name__}: {exc}"
            ) from exc
        if not exists:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
        try:
            info = client.get_collection(collection)
        except Exception as exc:
            raise VectorStoreUnavailable(
                f"读取 Qdrant collection 失败: {type(exc).__name__}: {exc}"
            ) from exc

        vectors = info.config.params.vectors
        existing_dimension = getattr(vectors, "size", None)
        if existing_dimension != dimension:
            raise VectorStoreUnavailable(
                f"Qdrant collection {collection} 维度为 {existing_dimension}，"
                f"当前模型维度为 {dimension}"
            )

        if collection in self._indexed_collections:
            return
        if cfg("qdrant_url").strip():
            for field_name, field_type in _index_fields:
                try:
                    client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=field_type,
                    )
                except Exception:
                    logger.debug(
                        "创建 Qdrant payload 索引失败: collection=%s field=%s",
                        collection,
                        field_name,
                        exc_info=True,
                    )
        self._indexed_collections.add(collection)

    def upsert_cards(self, cards: list[dict], *, model: str | None = None) -> int:
        """只为新增或内容变化的职位计算 Embedding 并写入 Qdrant。"""
        if not cards or not self.enabled:
            return 0
        from qdrant_client.models import PointStruct

        from analysis_work_content import embed_texts

        model = model or _model_name()
        collection = _collection_name(model)
        expected_chunks = {
            _job_id(card): _chunks(card)
            for card in cards
        }
        point_ids = {
            _chunk_point_id(job_id, chunk_index)
            for job_id, chunks in expected_chunks.items()
            for chunk_index in range(max(len(chunks), 1))
        }
        with self._lock:
            client = self._get_client()
            existing: dict[str, Any] = {}
            collection_exists = False
            try:
                collection_exists = client.collection_exists(collection)
                if collection_exists:
                    for point in client.retrieve(
                        collection_name=collection,
                        ids=list(point_ids),
                        with_vectors=False,
                        with_payload=True,
                    ):
                        if point.payload:
                            existing[str(point.id)] = point.payload
            except Exception as exc:
                raise VectorStoreUnavailable(
                    f"读取 Qdrant 增量索引失败: {type(exc).__name__}: {exc}"
                ) from exc

            pending = []
            for card in cards:
                job_id = _job_id(card)
                chunks = expected_chunks[job_id]
                content_hash = _content_hash(card)
                expected_ids = [
                    _chunk_point_id(job_id, chunk_index)
                    for chunk_index in range(len(chunks))
                ]
                anchor = existing.get(_chunk_point_id(job_id, 0), {})
                complete = bool(chunks) and (
                    anchor.get("content_hash") == content_hash
                    and anchor.get("chunk_count") == len(chunks)
                    and all(
                        existing.get(point_id, {}).get("content_hash")
                        == content_hash
                        for point_id in expected_ids
                    )
                )
                if not complete:
                    pending.append(card)
        if not pending:
            return 0

        pending_chunks = [
            (card, chunks)
            for card in pending
            if (chunks := _chunks(card))
        ]
        flat_chunks = [
            chunk.text
            for _, chunks in pending_chunks
            for chunk in chunks
        ]
        if not flat_chunks:
            if collection_exists:
                with self._lock:
                    client = self._get_client()
                    client.delete(
                        collection_name=collection,
                        points_selector=self._job_filter(pending),
                        wait=True,
                    )
            return len(pending)
        vectors = embed_texts(flat_chunks, model=model)
        if len(vectors) != len(flat_chunks):
            raise EmbeddingError("Embedding 返回数量与职位分块数量不一致")

        with self._lock:
            client = self._get_client()
            dimension = len(vectors[0]) if vectors else 0
            if not dimension:
                raise EmbeddingError("职位 Embedding 返回了空向量")
            if any(len(vector) != dimension for vector in vectors):
                raise EmbeddingError("职位 Embedding 维度不一致")
            self._ensure_collection(collection, dimension)

            client.delete(
                collection_name=collection,
                points_selector=self._job_filter(pending),
                wait=True,
            )
            points = []
            vector_index = 0
            for card, chunks in pending_chunks:
                job_id = _job_id(card)
                for chunk_index, chunk in enumerate(chunks):
                    points.append(
                        PointStruct(
                            id=_chunk_point_id(job_id, chunk_index),
                            vector=vectors[vector_index],
                            payload=_payload(
                                card,
                                job_id,
                                model,
                                chunk.text,
                                chunk.kind,
                                chunk_index,
                                len(chunks),
                            ),
                        )
                    )
                    vector_index += 1
            client.upsert(collection_name=collection, points=points, wait=True)
        return len(pending)

    @staticmethod
    def _job_filter(cards: list[dict]):
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        return Filter(
            must=[
                FieldCondition(
                    key="job_id",
                    match=MatchAny(any=[_job_id(card) for card in cards]),
                )
            ]
        )

    @staticmethod
    def _query_filter(cards: list[dict], search_params: Any = None):
        """限制检索范围和结构化硬条件，避免旧职位污染当前结果。"""
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        job_ids = [_job_id(card) for card in cards]
        conditions = [
            FieldCondition(key="job_id", match=MatchAny(any=job_ids))
        ]
        if search_params is not None:
            locations = [
                str(value).strip().rstrip("市")
                for value in (getattr(search_params, "location", None) or [])
                if str(value).strip()
            ]
            if (
                locations
                and "全国" not in locations
                and not getattr(search_params, "location_unlimited", False)
            ):
                conditions.append(
                    Filter(
                        should=[
                            FieldCondition(
                                key="city_name",
                                match=MatchAny(any=locations),
                            ),
                            FieldCondition(
                                key="city_name",
                                match=MatchValue(value=""),
                            ),
                        ],
                    )
                )
        return Filter(must=conditions)

    def search(
        self,
        cards: list[dict],
        query: str,
        *,
        search_params: Any = None,
        limit: int | None = None,
        threshold: float = default_match_threshold,
    ) -> list[tuple[int, float]]:
        """在当前职位集合内召回，返回与 ``match_jobs`` 相同的下标和分数。"""
        query_text = clean_text(query)[:max_query_chars]
        if not cards or not query_text or not self.enabled:
            return []
        from analysis_work_content import _query_vector

        model = _model_name()
        collection = _collection_name(model)
        try:
            query_vector = _query_vector(query_text, model)
            if not query_vector:
                raise EmbeddingError("查询文本 Embedding 失败")
            with self._lock:
                client = self._get_client()
                info = client.get_collection(collection)
                dimension = getattr(info.config.params.vectors, "size", None)
                if dimension != len(query_vector):
                    raise VectorStoreUnavailable(
                        f"Qdrant collection {collection} 维度不匹配"
                    )
                chunk_count = sum(len(_chunks(card)) for card in cards)
                response = client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    query_filter=self._query_filter(cards, search_params),
                    limit=max(chunk_count, 1),
                    with_payload=True,
                    score_threshold=threshold if threshold > 0 else None,
                )
                points = response.points or []
        except Exception as exc:
            if isinstance(exc, EmbeddingError):
                raise
            if isinstance(exc, VectorStoreUnavailable):
                raise
            raise VectorStoreUnavailable(
                f"Qdrant 查询失败: {type(exc).__name__}: {exc}"
            ) from exc

        index_by_job_id = {
            _job_id(card): index
            for index, card in enumerate(cards)
        }
        chunk_scores = []
        for point in points:
            if not point.payload:
                continue
            job_id = str(point.payload.get("job_id") or "")
            if job_id in index_by_job_id:
                chunk_scores.append((
                    index_by_job_id[job_id],
                    float(point.score),
                    str(point.payload.get("chunk_type") or "default"),
                ))
        ranked = aggregate_chunk_scores(
            chunk_scores,
            threshold=threshold,
            channel_count=2,
        )
        return ranked[:limit] if limit is not None else ranked

    def match_cards(
        self,
        cards: list[dict],
        query: str,
        *,
        search_params: Any = None,
        threshold: float = default_match_threshold,
    ) -> list[tuple[int, float]]:
        """索引当前职位并召回结果；调用方负责排除关键词。"""
        try:
            self.upsert_cards(cards)
            return self.search(
                cards,
                query,
                search_params=search_params,
                threshold=threshold,
            )
        except VectorStoreUnavailable:
            raise
        except EmbeddingError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailable(
                f"Qdrant 索引失败: {type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        """关闭 Qdrant 客户端，释放本地持久化文件锁。"""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
            self._indexed_collections.clear()


vector_store = QdrantJobStore()
