"""职位匹配、城市编码和筛选条件工具。"""

import re

from analysis_work_content import match_jobs


class CodeBook:
    """筛选条件与城市编码的统一入口。"""

    _options_file = "filter_options.json"
    _city_file = "city_codes.json"

    def __init__(self, data_dir=None):
        from pathlib import Path
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self._options = None
        self._cities = None

    def code_of(self, category, label):
        """将意图识别得到的选项名称转换为 JSON 中的筛选编码。

        这里故意只做精确匹配。模型输出的名称必须与
        ``data/filter_options.json`` 中的键一致，不能从用户原话猜测或
        拼接不存在的编码，避免把错误条件带入职位搜索 URL。
        """
        value = self._options_data().get(category, {}).get(label or "", "")
        return str(value) if value != "" else ""

    def codes_of(self, category, labels):
        """将多个意图选项按 JSON 码表转码，并以逗号连接供 URL 使用。"""
        if isinstance(labels, str):
            labels = [labels]
        codes = [self.code_of(category, label) for label in (labels or [])]
        return ",".join(code for code in codes if code)

    def labels_of(self, category):
        return list(self._options_data().get(category, {}))

    def city_code(self, name):
        normalized = (name or "").strip().rstrip("市")
        if not normalized:
            return ""
        for city, code in self._city_data().items():
            if city.rstrip("市") == normalized:
                return code
        return ""

    def city_names(self):
        return sorted(self._city_data(), key=len, reverse=True)

    def match_city(self, text, hint=""):
        names = self.city_names()
        normalized = (hint or "").strip().rstrip("市")
        if normalized:
            for name in names:
                if name.rstrip("市") == normalized:
                    return name
        for name in names:
            if name in text:
                return name
        return ""

    def _options_data(self):
        if self._options is None:
            self._options = self._read_json(self._options_file)
        return self._options

    def _city_data(self):
        if self._cities is None:
            self._cities = self._read_json(self._city_file)
        return self._cities

    def _read_json(self, filename):
        return __import__("json").loads((self._data_dir / filename).read_text(encoding="utf-8"))


code_book = CodeBook()


def deduplicate(values):
    """保持原顺序去重，并忽略空字符串。"""
    result = []
    seen = set()
    for value in values:
        value = value.strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def card_matches_exclusions(card, excluded_keywords):
    """判断职位卡片是否包含用户明确排除的工作内容。"""
    searchable = " ".join([
        str(card.get("jobName") or ""),
        str(card.get("postDescription") or ""),
        str(card.get("brandIndustry") or ""),
    ])
    searchable = re.sub(r"<[^>]+>", " ", searchable).casefold()
    return any(keyword.casefold() in searchable for keyword in excluded_keywords)


def match_card_batch(cards, query, excluded_keywords, search_params=None):
    """过滤一批职位并返回带相似度的匹配结果。

    启用 Qdrant 时优先使用向量库召回；Qdrant 不可用时回退到本地向量匹配。
    """
    eligible_cards = [
        card for card in cards
        if not card_matches_exclusions(card, excluded_keywords)
    ]
    if not eligible_cards:
        return []
    descriptions = [
        re.sub(r"<[^>]+>", " ", card.get("postDescription") or "").strip()
        for card in eligible_cards
    ]
    try:
        from vector_store import vector_store

        ranked = (
            vector_store.match_cards(
                eligible_cards,
                query,
                search_params=search_params,
                threshold=0.3,
            )
            if vector_store.enabled
            else match_jobs(descriptions, query, threshold=0.3)
        )
    except Exception as exc:
        from logging_config import get_logger

        get_logger(__name__).warning(
            "Qdrant 不可用，回退本地向量匹配: error=%s", exc
        )
        ranked = match_jobs(descriptions, query, threshold=0.3)
    return [
        {"job_card": eligible_cards[index], "score": score}
        for index, score in ranked
    ]
