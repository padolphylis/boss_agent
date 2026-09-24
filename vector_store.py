"""Qdrant 职位向量索引。

向量库只负责职位语义召回，SQLite 仍负责配置、会话和投递状态。
默认启用 Qdrant 本地持久化；设置 ``qdrant_enabled=false`` 可关闭。

* ``qdrant_url``：远程 Qdrant 地址，例如 ``http://127.0.0.1:6333``
* ``qdrant_api_key``：Qdrant Cloud 或受保护实例的 API Key
* ``qdrant_path``：未配置 URL 时的本地持久化目录
* ``qdrant_collection``：collection 前缀

每个 Embedding 模型使用独立 collection，避免不同维度的向量混用。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from config import get as cfg
from logging_config import get_logger

logger = get_logger(__name__)

_INDEX_FIELDS = (
    ("job_id", "keyword"),
)


class VectorStoreUnavailable(RuntimeError):
    """Qdrant 未安装、未配置或当前不可用。"""


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
    suffix = hashlib.sha1(f"{provider}\0{model}".encode("utf-8")).hexdigest()[:10]
    # Qdrant collection 名称不需要暴露模型原文，避免模型名包含斜杠等特殊字符。
    return f"{prefix}_{suffix}"


def _clean_text(value: Any) -> str:
    return re.sub(r"<[^>]+>", " ", str(value or "")).strip()


def _job_id(card: dict) -> str:
    value = str(card.get("encryptJobId") or "").strip()
    if value:
        return value
    raw = "|".join(
        [
            str(card.get("jobName") or ""),
            str(card.get("brandName") or ""),
            _clean_text(card.get("postDescription")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _point_id(job_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"boss-agent/job/{job_id}"))


def _content(card: dict) -> str:
    return _clean_text(
        " ".join(
            [
                str(card.get("jobName") or ""),
                str(card.get("brandName") or ""),
                str(card.get("brandIndustry") or ""),
                str(card.get("jobExperience") or ""),
                str(card.get("jobDegree") or ""),
                _clean_text(card.get("postDescription")),
            ]
        )
    )[:8000]


def _content_hash(card: dict) -> str:
    # Payload 字段变化也必须触发 upsert，否则城市、薪资等过滤元数据会过期。
    indexed_data = {
        "content": _content(card),
        "city_name": str(card.get("cityName") or "").strip().rstrip("市"),
        "city_code": str(card.get("city") or ""),
        "job_experience": str(card.get("jobExperience") or ""),
        "job_degree": str(card.get("jobDegree") or ""),
        "brand_scale_name": str(card.get("brandScaleName") or ""),
        "salary_desc": str(card.get("salaryDesc") or ""),
    }
    serialized = json.dumps(indexed_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _payload(card: dict, job_id: str, model: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "job_name": str(card.get("jobName") or ""),
        "brand_name": str(card.get("brandName") or ""),
        "city_name": str(card.get("cityName") or "").strip().rstrip("市"),
        "city_code": str(card.get("city") or ""),
        "job_experience": str(card.get("jobExperience") or ""),
        "job_degree": str(card.get("jobDegree") or ""),
        "brand_scale_name": str(card.get("brandScaleName") or ""),
        "salary_desc": str(card.get("salaryDesc") or ""),
        "content_hash": _content_hash(card),
        "embedding_model": model,
    }


class QdrantJobStore:
    """职位向量的增量索引和带元数据过滤的召回。"""

    def __init__(self, client=None):
        self._client = client
        self._lock = RLock()
        self._indexed_collections: set[str] = set()
        # 测试和故障隔离可以显式覆盖配置开关；None 表示读取配置。
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
            info = client.get_collection(collection)
        except Exception:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            info = client.get_collection(collection)

        vectors = info.config.params.vectors
        existing_dimension = getattr(vectors, "size", None)
        if existing_dimension != dimension:
            raise VectorStoreUnavailable(
                f"Qdrant collection {collection} 维度为 {existing_dimension}，"
                f"当前模型维度为 {dimension}"
            )

        if collection in self._indexed_collections:
            return
        # Local mode is intended for small single-process indexes; its payload
        # indexes are no-ops. Create indexes for server-backed deployments.
        if cfg("qdrant_url").strip():
            for field_name, field_type in _INDEX_FIELDS:
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
        client = self._get_client()
        point_ids = [_point_id(_job_id(card)) for card in cards]
        with self._lock:
            existing: dict[str, Any] = {}
            try:
                for point in client.retrieve(
                    collection_name=collection,
                    ids=point_ids,
                    with_vectors=False,
                    with_payload=True,
                ):
                    if point.payload:
                        existing[str(point.id)] = point.payload
            except Exception:
                # collection 尚不存在时，后续创建流程会一次性索引全部职位。
                existing = {}

            pending = [
                card for card in cards
                if (
                    existing.get(_point_id(_job_id(card)), {}).get("content_hash")
                    != _content_hash(card)
                )
            ]
        if not pending:
            return 0

        vectors = embed_texts([_content(card) for card in pending], model=model)
        valid = [
            (card, vector)
            for card, vector in zip(pending, vectors)
            if vector
        ]
        if len(valid) != len(pending):
            raise VectorStoreUnavailable(
                "部分职位 Embedding 失败，回退本地向量匹配"
            )
        if not valid:
            return 0

        with self._lock:
            dimension = len(valid[0][1])
            if any(len(vector) != dimension for _, vector in valid):
                raise VectorStoreUnavailable("职位 Embedding 维度不一致")
            self._ensure_collection(collection, dimension)
            points = [
                PointStruct(
                    id=_point_id(_job_id(card)),
                    vector=vector,
                    payload=_payload(card, _job_id(card), model),
                )
                for card, vector in valid
            ]
            client.upsert(collection_name=collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        cards: list[dict],
        query: str,
        *,
        search_params: Any = None,
        limit: int | None = None,
        threshold: float = 0.3,
    ) -> list[tuple[int, float]]:
        """在当前职位集合内召回，返回与 ``match_jobs`` 相同的下标和分数。"""
        if not cards or not query.strip() or not self.enabled:
            return []
        from analysis_work_content import _query_vector, cosine_similarity

        model = _model_name()
        collection = _collection_name(model)
        try:
            query_vector = _query_vector(query.strip(), model)
            if not query_vector:
                raise VectorStoreUnavailable("查询文本 Embedding 失败")
            client = self._get_client()
            info = client.get_collection(collection)
            dimension = getattr(info.config.params.vectors, "size", None)
            if dimension != len(query_vector):
                raise VectorStoreUnavailable(
                    f"Qdrant collection {collection} 维度不匹配"
                )
            # 本地 Qdrant 的过滤查询当前只稳定返回第一条结果。
            # 职位批次通常较小，因此按本批 ID 精确取出向量后本地排序。
            points = client.retrieve(
                collection_name=collection,
                ids=[_point_id(_job_id(card)) for card in cards],
                with_vectors=True,
                with_payload=True,
            )
        except Exception as exc:
            if isinstance(exc, VectorStoreUnavailable):
                raise
            raise VectorStoreUnavailable(
                f"Qdrant 查询失败: {type(exc).__name__}: {exc}"
            ) from exc

        scores: dict[str, float] = {}
        for point in points:
            if not point.payload or not point.vector:
                continue
            score = round(cosine_similarity(query_vector, list(point.vector)), 4)
            if score >= threshold:
                scores[str(point.payload.get("job_id"))] = score
        ranked = [
            (index, scores[_job_id(card)])
            for index, card in enumerate(cards)
            if _job_id(card) in scores
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit] if limit is not None else ranked

    def match_cards(
        self,
        cards: list[dict],
        query: str,
        *,
        search_params: Any = None,
        threshold: float = 0.3,
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
