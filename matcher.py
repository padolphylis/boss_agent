"""职位匹配、城市编码和筛选条件工具。"""

import json
import re
from pathlib import Path

from analysis_work_content import match_jobs


class CodeBook:
    """筛选条件与城市编码的统一入口。"""

    _city_file = "city_codes.json"
    # 每个筛选项使用独立码表，避免一个文件同时承担多种含义。
    _category_files = {
        "salary": "money_codes.json",
        "experience": "experience_codes.json",
        "degree": "degree_codes.json",
        "scale": "scale_codes.json",
    }

    def __init__(self, data_dir=None):
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self._cities = None
        self._options = {}

    def code_of(self, category, label):
        """将意图识别得到的选项名称转换为对应码表中的筛选编码。

        所有筛选项都只做精确匹配。自然语言到码表选项的语义映射
        由意图模型完成，后端这里只负责读取 data/*_codes.json。
        """
        value = self._category_data(category).get(str(label or "").strip(), "")
        return str(value) if value != "" else ""

    def codes_of(self, category, labels):
        """将多个意图选项按 JSON 码表转码，并以逗号连接供 URL 使用。"""
        if isinstance(labels, str):
            labels = [labels]
        codes = [self.code_of(category, label) for label in (labels or [])]
        return ",".join(code for code in codes if code)

    def labels_of(self, category):
        return list(self._category_data(category))

    def filename_of(self, category):
        """返回分类对应的码表文件名，供校验错误和提示信息使用。"""
        return self._category_files.get(category, "")

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

    def _category_data(self, category):
        if category not in self._options:
            filename = self._category_files.get(category)
            self._options[category] = (
                self._read_json(filename) if filename else {}
            )
        return self._options[category]

    def _city_data(self):
        if self._cities is None:
            self._cities = self._read_json(self._city_file)
        return self._cities

    def _read_json(self, filename):
        return json.loads((self._data_dir / filename).read_text(encoding="utf-8"))


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
