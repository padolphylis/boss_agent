"""职位内容向量化与相似度计算。"""

import re
from collections import OrderedDict
from time import sleep
from threading import Lock

import openai

from config import get as cfg
from logging_config import get_logger, fingerprint

logger = get_logger(__name__)

_QUERY_VECTOR_CACHE: OrderedDict[tuple[str, str, str, str], list[float]] = OrderedDict()
_QUERY_VECTOR_CACHE_LOCK = Lock()
_QUERY_VECTOR_CACHE_SIZE = 128


def _client() -> openai.OpenAI:
    # Embedding 与对话模型使用完全独立的凭据；旧配置作为迁移兼容回退。
    kwargs: dict = {
        "api_key": cfg("embedding_openai_api_key") or cfg("openai_api_key")
    }
    base_url = cfg("embedding_openai_base_url") or cfg("openai_base_url")
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text).strip()


def embed_texts(
    texts: list[str],
    *,
    batch_size: int = 16,
    delay: float = 0.1,
    model: str | None = None,
) -> list[list[float]]:
    """批量文本向量化，返回与 texts 等长的向量列表。"""
    client = _client()
    model = model or cfg("embedding_openai_model") or cfg(
        "embedding_model", "text-embedding-3-small"
    )
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.debug(
            "向量化请求: model=%s batch_start=%s batch_size=%s texts(%s)",
            model,
            i,
            len(batch),
            fingerprint("".join(batch)),
        )
        try:
            resp = client.embeddings.create(model=model, input=batch)
            all_vecs.extend(item.embedding for item in resp.data)
            logger.debug(
                "向量化响应: model=%s batch_start=%s 返回条数=%s 维度=%s",
                model,
                i,
                len(resp.data),
                len(resp.data[0].embedding) if resp.data else 0,
            )
        except Exception as exc:
            logger.exception(
                "向量化失败: batch_start=%s batch_size=%s error=%s",
                i,
                len(batch),
                f"{type(exc).__name__}: {exc}",
            )
            all_vecs.extend([] for _ in batch)
        if i + batch_size < len(texts):
            sleep(delay)

    return all_vecs


def _query_vector(query: str, model: str) -> list[float]:
    """缓存相同模型和需求文本的查询向量，避免每个职位批次重复请求。"""
    api_key = cfg("embedding_openai_api_key") or cfg("openai_api_key") or ""
    key = (
        model,
        str(cfg("embedding_openai_base_url") or cfg("openai_base_url") or ""),
        fingerprint(api_key),
        query,
    )
    with _QUERY_VECTOR_CACHE_LOCK:
        cached = _QUERY_VECTOR_CACHE.get(key)
        if cached is not None:
            _QUERY_VECTOR_CACHE.move_to_end(key)
            return cached

    vectors = embed_texts([query], model=model)
    vector = vectors[0] if vectors else []
    if vector:
        with _QUERY_VECTOR_CACHE_LOCK:
            _QUERY_VECTOR_CACHE[key] = vector
            _QUERY_VECTOR_CACHE.move_to_end(key)
            while len(_QUERY_VECTOR_CACHE) > _QUERY_VECTOR_CACHE_SIZE:
                _QUERY_VECTOR_CACHE.popitem(last=False)
    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def match_jobs(
    descriptions: list[str],
    query: str,
    *,
    threshold: float = 0.0,
    batch_size: int = 16,
) -> list[tuple[int, float]]:
    """将职位描述与用户需求做相似度匹配。

    参数:
        descriptions: 每个职位的纯文本描述（调用方负责清洗 HTML）。
        query: 用户求职需求。
        threshold: 最低分数，低于此值的不返回。
        batch_size: 每批向量化条数，由前端设置。

    返回:
        [(原始下标, 相似度分数), ...] 按分数降序。
    """
    if not descriptions or not query.strip():
        logger.debug("跳过职位匹配: descriptions=%s query_empty=%s", len(descriptions), not query.strip())
        return []

    valid = [(i, t[:8000]) for i, t in enumerate(descriptions) if t.strip()]
    if not valid:
        logger.warning("职位描述清洗后为空: total=%s", len(descriptions))
        return []

    model = cfg("embedding_openai_model") or cfg(
        "embedding_model", "text-embedding-3-small"
    )
    query_vec = _query_vector(query.strip(), model)
    if not query_vec:
        logger.warning("查询文本向量化失败，无法进行职位匹配")
        return []

    job_vecs = embed_texts(
        [t for _, t in valid],
        batch_size=batch_size,
        model=model,
    )

    results = []
    for (idx, _), vec in zip(valid, job_vecs):
        if not vec:
            continue
        score = cosine_similarity(query_vec, vec)
        if score >= threshold:
            results.append((idx, round(score, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        "职位匹配完成: descriptions=%s valid=%s matched=%s threshold=%.2f",
        len(descriptions),
        len(valid),
        len(results),
        threshold,
    )
    return results
