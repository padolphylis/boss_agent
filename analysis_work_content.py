from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from threading import Lock
from time import sleep
from typing import Literal, NamedTuple

import openai

from config import get as cfg
from logging_config import fingerprint, get_logger

logger = get_logger(__name__)

_query_vector_cache: OrderedDict[tuple[str, str, str, str], list[float]] = OrderedDict()
_query_vector_cache_lock = Lock()
_query_vector_cache_size = 128
embedding_retries = 3
embedding_concurrency = 3
max_document_chars = 8000
max_query_chars = 12000
default_chunk_size = 1200
default_chunk_overlap = 150
rrf_k = 60
rrf_max_contributions = 3
default_match_threshold = 0.45
full_recall_threshold = -1.0


class EmbeddingError(RuntimeError):
    """Embedding 服务不可用或返回了不完整的向量。"""

    error_source = "embedding"


class JobChunk(NamedTuple):
    """带来源类型的职位索引块。"""

    kind: Literal["summary", "description"]
    text: str


def _client() -> openai.OpenAI:
    kwargs: dict = {
        "api_key": cfg("embedding_openai_api_key") or cfg("openai_api_key")
    }
    base_url = cfg("embedding_openai_base_url") or cfg("openai_base_url")
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def clean_text(text: str) -> str:
    """清洗网页文本，同时保留段落边界供后续分块。"""
    text = html.unescape(str(text or ""))
    text = re.sub(
        r"<\s*/?\s*(?:br|p|div|li|section|article|h[1-6]|ul|ol|tr)"
        r"\b[^>]*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_benefits(value) -> list[str]:
    """将职位列表或详情 DOM 中的福利字段规范化为去重后的文本列表。"""
    if isinstance(value, str):
        values = re.split(r"[,，、;；|\n]+", value)
    elif isinstance(value, set):
        values = sorted(value, key=str)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    elif value:
        values = [value]
    else:
        values = []

    benefits = []
    seen = set()
    for item in values:
        if isinstance(item, dict):
            item = (
                item.get("name")
                or item.get("label")
                or item.get("value")
                or item.get("tagName")
                or item.get("title")
                or item.get("text")
                or ""
            )
        text = clean_text(item)
        if text in {"...", "…"}:
            continue
        if text and text not in seen:
            benefits.append(text)
            seen.add(text)
    return benefits


def merge_benefits(*values) -> list[str]:
    """合并多个福利来源，保持首次出现顺序并去重。"""
    benefits = []
    seen = set()
    for value in values:
        for benefit in normalize_benefits(value):
            if benefit not in seen:
                benefits.append(benefit)
                seen.add(benefit)
    return benefits


def job_benefits(card: dict) -> list[str]:
    """合并职位卡片和详情中可能出现的福利字段并保持顺序去重。"""
    return merge_benefits(
        card.get("welfareList"),
        card.get("benefitList"),
        card.get("benefits"),
    )


def job_city(card: dict) -> str:
    """返回可用于展示、过滤和索引的城市名称，不把数字编码当作名称。"""
    city_name = clean_text(card.get("cityName") or "").rstrip("市")
    if city_name:
        return city_name
    city = clean_text(card.get("city") or "").rstrip("市")
    return "" if city.isdigit() else city


def build_job_summary(card: dict) -> str:
    """构造不参与长文本切分的职位结构化摘要。"""
    benefits = job_benefits(card)
    fields = (
        ("职位名称", card.get("jobName")),
        ("公司", card.get("brandName") or card.get("brandComName")),
        ("行业", card.get("brandIndustry")),
        ("工作地点", job_city(card)),
        ("薪资", card.get("salaryDesc") or card.get("salary")),
        ("经验要求", card.get("jobExperience")),
        ("学历要求", card.get("jobDegree")),
        ("公司规模", card.get("brandScaleName")),
        ("福利待遇", "、".join(benefits)),
    )
    lines = [
        f"{label}：{value}"
        for label, raw_value in fields
        if (value := clean_text(str(raw_value or "")))
    ]
    return "\n".join(lines)[:max_document_chars]


def chunk_text(
    text: str,
    *,
    chunk_size: int = default_chunk_size,
    overlap: int = default_chunk_overlap,
) -> list[str]:
    """按段落和句子边界切分文本，过长单段再使用滑动窗口兜底。"""
    text = clean_text(text)
    chunk_size = max(int(chunk_size), 1)
    overlap = max(0, min(int(overlap), chunk_size - 1))
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    units = [
        unit.strip()
        for unit in re.split(r"(?<=[。！？；：.!?;:\n])", text)
        if unit.strip()
    ]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            step = chunk_size - overlap
            for start in range(0, len(unit), step):
                piece = unit[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_size >= len(unit):
                    break
            continue

        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            prefix = current[-overlap:] if overlap else ""
            candidate = f"{prefix}\n{unit}".strip() if prefix else unit
            current = candidate if len(candidate) <= chunk_size else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_job_chunk_records(card: dict) -> list[JobChunk]:
    """摘要保持为单块，只有职位描述按段落和句子边界切分。"""
    chunks = []
    summary = build_job_summary(card)
    if summary:
        chunks.append(JobChunk("summary", summary))

    description = clean_text(card.get("postDescription"))[:max_document_chars]
    chunks.extend(
        JobChunk("description", text)
        for text in chunk_text(description)
    )
    return chunks


def embed_texts(
    texts: list[str],
    *,
    batch_size: int = 16,
    delay: float = 0.1,
    model: str | None = None,
    retries: int = embedding_retries,
    concurrency: int = embedding_concurrency,
) -> list[list[float]]:
    """批量文本向量化，失败时重试并抛出明确异常。"""
    if not texts:
        return []

    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size 必须是正整数") from exc
    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    try:
        delay = max(float(delay), 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("delay 必须是非负数字") from exc
    try:
        concurrency = int(concurrency)
    except (TypeError, ValueError) as exc:
        raise ValueError("concurrency 必须是正整数") from exc
    if concurrency <= 0:
        raise ValueError("concurrency 必须是正整数")

    client = _client()
    model = model or cfg("embedding_openai_model") or cfg(
        "embedding_model", "text-embedding-3-small"
    )
    attempts = max(int(retries), 1)

    batches = [
        (i, texts[i:i + batch_size])
        for i in range(0, len(texts), batch_size)
    ]

    def embed_batch(item: tuple[int, list[str]]) -> list[list[float]]:
        batch_start, batch = item
        logger.debug(
            "向量化请求: model=%s batch_start=%s batch_size=%s texts(%s)",
            model,
            batch_start,
            len(batch),
            fingerprint("".join(batch)),
        )
        for attempt in range(1, attempts + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                data = list(resp.data or [])
                if len(data) != len(batch):
                    raise EmbeddingError(
                        f"Embedding 返回数量不一致: expected={len(batch)} actual={len(data)}"
                    )
                indices = [getattr(item, "index", None) for item in data]
                if (
                    all(isinstance(index, int) for index in indices)
                    and sorted(indices) == list(range(len(data)))
                ):
                    ordered = [None] * len(data)
                    for index, item in zip(indices, data):
                        ordered[index] = item
                    data = ordered
                vectors = [list(item.embedding or []) for item in data]
                if any(not vector for vector in vectors):
                    raise EmbeddingError("Embedding 返回了空向量")
                logger.debug(
                    "向量化响应: model=%s batch_start=%s 返回条数=%s 维度=%s",
                    model,
                    batch_start,
                    len(vectors),
                    len(vectors[0]) if vectors else 0,
                )
                return vectors
            except Exception as exc:
                if attempt >= attempts:
                    logger.exception(
                        "向量化失败: model=%s batch_start=%s batch_size=%s attempts=%s",
                        model,
                        batch_start,
                        len(batch),
                        attempts,
                    )
                    if isinstance(exc, EmbeddingError):
                        raise
                    raise EmbeddingError(
                        f"Embedding 请求失败: {type(exc).__name__}"
                    ) from exc
                logger.warning(
                    "向量化暂时失败，准备重试: model=%s batch_start=%s attempt=%s/%s error=%s",
                    model,
                    batch_start,
                    attempt,
                    attempts,
                    type(exc).__name__,
                )
                sleep(min(2.0, delay * (2 ** (attempt - 1))))

        raise EmbeddingError(
            f"Embedding 请求失败: batch_start={batch_start}"
        )

    with ThreadPoolExecutor(
        max_workers=min(concurrency, len(batches)),
        thread_name_prefix="embedding",
    ) as executor:
        vectors_by_batch = executor.map(embed_batch, batches)
        return [
            vector
            for batch_vectors in vectors_by_batch
            for vector in batch_vectors
        ]


def _query_vector(query: str, model: str) -> list[float]:
    """缓存相同模型和需求文本的查询向量，避免每个职位批次重复请求。"""
    api_key = cfg("embedding_openai_api_key") or cfg("openai_api_key") or ""
    key = (
        model,
        str(cfg("embedding_openai_base_url") or cfg("openai_base_url") or ""),
        fingerprint(api_key),
        query,
    )
    with _query_vector_cache_lock:
        cached = _query_vector_cache.get(key)
        if cached is not None:
            _query_vector_cache.move_to_end(key)
            return cached

    vectors = embed_texts([query], model=model)
    vector = vectors[0] if vectors else []
    if vector:
        with _query_vector_cache_lock:
            _query_vector_cache[key] = vector
            _query_vector_cache.move_to_end(key)
            while len(_query_vector_cache) > _query_vector_cache_size:
                _query_vector_cache.popitem(last=False)
    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def aggregate_chunk_scores(
    chunk_scores: list[tuple[int, float]],
    *,
    threshold: float = 0.0,
    channel_count: int | None = None,
) -> list[tuple[int, float]]:
    """按摘要/描述通道的 chunk 排名做归一化 reciprocal-rank fusion。"""
    channel_scores: dict[str, list[tuple[int, float]]] = {}
    all_channels = set()
    for item in chunk_scores:
        if len(item) == 2:
            document_index, score = item
            channel = "default"
        else:
            document_index, score, channel = item
            channel = str(channel or "default")
        score = float(score)
        all_channels.add(channel)
        if score >= threshold:
            channel_scores.setdefault(channel, []).append(
                (document_index, score),
            )

    if not channel_scores:
        return []
    expected_channels = max(
        int(channel_count or 0),
        len(all_channels),
        1,
    )
    denominator = expected_channels / (rrf_k + 1)
    fused_scores: dict[int, float] = {}
    for scores in channel_scores.values():
        ranked_channel = sorted(
            scores,
            key=lambda item: (-item[1], item[0]),
        )
        contributions: dict[int, int] = {}
        for rank, (document_index, _score) in enumerate(ranked_channel, 1):
            contribution_count = contributions.get(document_index, 0)
            if contribution_count >= rrf_max_contributions:
                continue
            contributions[document_index] = contribution_count + 1
            fused_scores[document_index] = (
                fused_scores.get(document_index, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    ranked = [
        (index, round(min(score / denominator, 1.0), 4))
        for index, score in fused_scores.items()
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def match_jobs(
    descriptions: list[str],
    query: str,
    *,
    threshold: float = 0.0,
    batch_size: int = 16,
) -> list[tuple[int, float]]:
    """将纯文本职位文档分块后与用户需求做相似度匹配。"""
    chunk_groups: list[list[str]] = []
    for text in descriptions:
        cleaned = clean_text(text)[:max_document_chars]
        chunk_groups.append(chunk_text(cleaned))
    return match_job_chunks(
        chunk_groups,
        query,
        threshold=threshold,
        batch_size=batch_size,
    )


def match_job_chunks(
    chunk_groups: list[list[str]],
    query: str,
    *,
    threshold: float = 0.0,
    batch_size: int = 16,
) -> list[tuple[int, float]]:
    """匹配职位 chunk；支持 JobChunk 以保留摘要/描述通道供 RRF 融合。"""
    query_text = clean_text(query)[:max_query_chars]
    if not chunk_groups or not query_text:
        logger.debug(
            "跳过职位匹配: jobs=%s query_empty=%s",
            len(chunk_groups),
            not query_text,
        )
        return []

    valid = []
    all_channels = set()
    for index, chunks in enumerate(chunk_groups):
        records = []
        for chunk in chunks:
            if isinstance(chunk, JobChunk):
                channel, text = chunk.kind, chunk.text
            elif isinstance(chunk, tuple) and len(chunk) == 2:
                channel, text = str(chunk[0]), chunk[1]
            else:
                channel, text = "default", chunk
            cleaned = clean_text(text)
            if cleaned:
                records.append((channel, cleaned))
                all_channels.add(channel)
        if records:
            valid.append((index, records))
    if not valid:
        logger.warning("职位内容清洗后为空: total=%s", len(chunk_groups))
        return []

    model = cfg("embedding_openai_model") or cfg(
        "embedding_model", "text-embedding-3-small"
    )
    query_vec = _query_vector(query_text, model)
    if not query_vec:
        raise EmbeddingError("查询文本向量化失败")

    flat_chunks = [
        text
        for _, records in valid
        for _, text in records
    ]
    chunk_owners = [
        index
        for index, records in valid
        for _ in records
    ]
    chunk_channels = [
        channel
        for _, records in valid
        for channel, _ in records
    ]
    chunk_vecs = embed_texts(
        flat_chunks,
        batch_size=batch_size,
        model=model,
    )
    if len(chunk_vecs) != len(flat_chunks):
        raise EmbeddingError(
            f"职位 chunk Embedding 返回数量不一致: "
            f"expected={len(flat_chunks)} actual={len(chunk_vecs)}"
        )

    chunk_scores = []
    for index, channel, vec in zip(chunk_owners, chunk_channels, chunk_vecs):
        if not vec:
            continue
        if len(vec) != len(query_vec):
            raise EmbeddingError(
                f"Embedding 维度不一致: query={len(query_vec)} document={len(vec)}"
            )
        score = cosine_similarity(query_vec, vec)
        chunk_scores.append((index, score, channel))
    results = aggregate_chunk_scores(
        chunk_scores,
        threshold=threshold,
        channel_count=(
            2
            if all_channels.intersection({"summary", "description"})
            else len(all_channels)
        ),
    )
    logger.info(
        "职位匹配完成: jobs=%s valid=%s chunks=%s matched=%s threshold=%.2f",
        len(chunk_groups),
        len(valid),
        len(flat_chunks),
        len(results),
        threshold,
    )
    return results
