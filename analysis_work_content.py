"""职位内容向量化与相似度计算。"""

import re
from time import sleep

import openai

from config import get as cfg


def _client() -> openai.OpenAI:
    kwargs: dict = {"api_key": cfg("openai_api_key")}
    base_url = cfg("openai_base_url")
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
    model = model or cfg("embedding_model", "text-embedding-3-small")
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            all_vecs.extend(item.embedding for item in resp.data)
        except Exception as exc:
            print(f"向量化失败: {type(exc).__name__}: {exc}")
            all_vecs.extend([] for _ in batch)
        if i + batch_size < len(texts):
            sleep(delay)

    return all_vecs


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
        return []

    valid = [(i, t[:8000]) for i, t in enumerate(descriptions) if t.strip()]
    if not valid:
        return []

    query_vec = embed_texts([query])[0]
    if not query_vec:
        return []

    job_vecs = embed_texts(
        [t for _, t in valid],
        batch_size=batch_size,
    )

    results = []
    for (idx, _), vec in zip(valid, job_vecs):
        if not vec:
            continue
        score = cosine_similarity(query_vec, vec)
        if score >= threshold:
            results.append((idx, round(score, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
