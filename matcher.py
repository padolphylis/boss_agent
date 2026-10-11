import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from analysis_work_content import (
    default_match_threshold,
    clean_text,
    job_city,
    match_jobs as _match_text_jobs,
    match_job_chunks as _match_job_chunks,
)


class CodeBook:
    """筛选条件与城市编码的统一入口。"""

    _city_file = "city_codes.json"
    _category_files: ClassVar[dict[str, str]] = {
        "salary": "money_codes.json",
        "job_type": "job_type_codes.json",
        "experience": "experience_codes.json",
        "degree": "degree_codes.json",
        "scale": "scale_codes.json",
    }

    def __init__(self, data_dir=None):
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self._cities = None
        self._options = {}

    def code_of(self, category, label):
        """将意图识别得到的选项名称转换为对应码表中的筛选编码。"""
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


@dataclass(frozen=True)
class SalaryRange:
    """统一使用“千元/月”表示的薪资范围；None 表示开放边界。"""

    minimum: float | None
    maximum: float | None


_salary_amount = r"\d+(?:\.\d+)?"
_salary_range_pattern = re.compile(
    rf"(?P<minimum>{_salary_amount})\s*"
    rf"(?P<minimum_unit>K|k|千|万|元)?\s*"
    r"(?:-|－|—|–|~|～|至|到)\s*"
    rf"(?P<maximum>{_salary_amount})\s*"
    rf"(?P<maximum_unit>K|k|千|万|元)?",
    re.IGNORECASE,
)
_salary_bound_pattern = re.compile(
    rf"(?P<amount>{_salary_amount})\s*"
    rf"(?P<unit>K|k|千|万|元)?\s*"
    r"(?P<bound>以上|起步|起|至少|不少于|下限|以下|以内|最多|不超过|上限)",
    re.IGNORECASE,
)
_salary_single_pattern = re.compile(
    rf"(?P<amount>{_salary_amount})\s*(?P<unit>K|k|千|万|元)",
    re.IGNORECASE,
)
_salary_context_single_pattern = re.compile(
    rf"(?P<amount>{_salary_amount})\s*(?P<unit>K|k|千|万|元)?",
    re.IGNORECASE,
)
_salary_context_pattern = re.compile(r"(?:薪资|薪水|工资|月薪)")
_salary_unlimited_pattern = re.compile(
    r"(?:薪资|薪水|工资|月薪).{0,8}(?:不限|不限制|随意|都可以|无要求|没有要求)"
)

_platform_salary_ranges = (
    ("3K以下", SalaryRange(None, 3.0)),
    ("3-5K", SalaryRange(3.0, 5.0)),
    ("5-10K", SalaryRange(5.0, 10.0)),
    ("10-20K", SalaryRange(10.0, 20.0)),
    ("20-50K", SalaryRange(20.0, 50.0)),
    ("50K以上", SalaryRange(50.0, None)),
)


def _salary_unit_multiplier(unit: str | None) -> float | None:
    normalized = str(unit or "").strip().casefold()
    if normalized in {"k", "千"}:
        return 1.0
    if normalized == "万":
        return 10.0
    if normalized == "元":
        return 0.001
    return None


def _salary_amount(value: str, unit: str | None) -> float | None:
    multiplier = _salary_unit_multiplier(unit)
    if multiplier is None:
        amount = float(value)
        multiplier = 0.001 if amount >= 1000 else 1.0
    return round(float(value) * multiplier, 4)


def parse_salary_range(value: object) -> SalaryRange | None:
    """解析职位或用户薪资文本，返回千元/月范围。"""
    text = re.sub(r"\s+", "", str(value or ""))
    if not text or text.casefold() in {"不限", "面议", "薪资面议"}:
        return None
    if re.search(r"(?:元/天|元每天|日薪|每天)", text, re.IGNORECASE):
        return None

    range_match = _salary_range_pattern.search(text)
    if range_match and (
        range_match.group("minimum_unit")
        or range_match.group("maximum_unit")
    ):
        minimum_unit = range_match.group("minimum_unit")
        maximum_unit = range_match.group("maximum_unit") or minimum_unit
        minimum = _salary_amount(
            range_match.group("minimum"),
            minimum_unit or maximum_unit,
        )
        maximum = _salary_amount(
            range_match.group("maximum"),
            maximum_unit,
        )
        if minimum is not None and maximum is not None:
            return SalaryRange(minimum, maximum) if minimum <= maximum else None

    bound_match = _salary_bound_pattern.search(text)
    if bound_match and bound_match.group("unit"):
        amount = _salary_amount(
            bound_match.group("amount"),
            bound_match.group("unit"),
        )
        if amount is not None:
            bound = bound_match.group("bound")
            if bound in {"以上", "起步", "起", "至少", "不少于", "下限"}:
                return SalaryRange(amount, None)
            return SalaryRange(None, amount)

    single_match = _salary_single_pattern.search(text)
    if single_match and single_match.group("unit"):
        amount = _salary_amount(
            single_match.group("amount"),
            single_match.group("unit"),
        )
        if amount is not None:
            return SalaryRange(amount, amount)
    return None


def _format_salary_amount(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def normalize_salary_requirements(values) -> list[str]:
    """规范化用户真实薪资，并拒绝无法执行硬过滤的表达。"""
    if isinstance(values, str):
        values = [values]
    result = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in {"不限", "不设限"}:
            continue
        parsed = parse_salary_range(text)
        if parsed is None:
            raise ValueError(
                f"salary 无法解析为月薪范围: {text}；请使用如 7-8K、7K以上的表达"
            )
        if parsed.minimum is None:
            canonical = f"{_format_salary_amount(parsed.maximum)}K以下"
        elif parsed.maximum is None:
            canonical = f"{_format_salary_amount(parsed.minimum)}K以上"
        elif parsed.minimum == parsed.maximum:
            canonical = f"{_format_salary_amount(parsed.minimum)}K"
        else:
            canonical = (
                f"{_format_salary_amount(parsed.minimum)}-"
                f"{_format_salary_amount(parsed.maximum)}K"
            )
        if canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


def extract_salary_requirements(text: str) -> list[str]:
    """从用户原话提取明确薪资，防止结构化模型漏掉真实区间。"""
    normalized_text = str(text or "")
    candidates: list[str] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in _salary_range_pattern.finditer(normalized_text):
        if match.group("minimum_unit") or match.group("maximum_unit"):
            candidates.append(match.group(0))
            occupied_spans.append(match.span())

    for match in _salary_bound_pattern.finditer(normalized_text):
        if match.group("unit"):
            candidates.append(match.group(0))
            occupied_spans.append(match.span())

    for match in _salary_single_pattern.finditer(normalized_text):
        if any(
            start <= match.start() and match.end() <= end
            for start, end in occupied_spans
        ):
            continue
        candidates.append(match.group(0))

    if candidates:
        return normalize_salary_requirements(candidates)

    for context in _salary_context_pattern.finditer(normalized_text):
        fragment = re.split(
            r"[，,；;。\n]",
            normalized_text[context.end():],
            maxsplit=1,
        )[0][:32]
        range_match = _salary_range_pattern.search(fragment)
        bound_match = _salary_bound_pattern.search(fragment)
        single_match = _salary_context_single_pattern.search(fragment)
        match = range_match or bound_match or single_match
        if not match:
            continue

        if range_match is not None and match is range_match:
            minimum_unit = range_match.group("minimum_unit")
            maximum_unit = range_match.group("maximum_unit")
            if not minimum_unit and not maximum_unit:
                candidate = (
                    f"{range_match.group('minimum')}K-"
                    f"{range_match.group('maximum')}K"
                )
            else:
                candidate = range_match.group(0)
        elif bound_match is not None and match is bound_match:
            candidate = (
                f"{bound_match.group('amount')}"
                f"{bound_match.group('unit') or 'K'}"
                f"{bound_match.group('bound')}"
            )
        else:
            candidate = (
                f"{single_match.group('amount')}"
                f"{single_match.group('unit') or 'K'}"
            )
        candidates.append(candidate)

    return normalize_salary_requirements(candidates) if candidates else []


def salary_is_unlimited(text: str) -> bool:
    """判断用户是否明确取消薪资限制。"""
    return bool(_salary_unlimited_pattern.search(str(text or "")))


def salary_filter_buckets(values) -> list[str]:
    """把真实薪资转换为 BOSS 搜索所需的、可能扩大召回的薪资桶。"""
    buckets = []
    seen = set()
    requirements = normalize_salary_requirements(values)
    for requirement in requirements:
        expected = parse_salary_range(requirement)
        if expected is None:
            continue
        for label, platform in _platform_salary_ranges:
            if _salary_ranges_overlap(expected, platform):
                if label not in seen:
                    buckets.append(label)
                    seen.add(label)
    return buckets


def card_matches_salary(card, search_params=None, *, allow_unknown=False) -> bool:
    """按用户真实薪资判断职位薪资范围是否存在交集。"""
    requirements = getattr(search_params, "salary", None) if search_params else None
    if not requirements:
        return True
    expected_ranges = [
        parse_salary_range(value)
        for value in requirements
    ]
    expected_ranges = [item for item in expected_ranges if item is not None]
    if not expected_ranges:
        return False

    advertised = parse_salary_range(
        card.get("salaryDesc") or card.get("salary")
    )
    if advertised is None:
        return bool(allow_unknown)
    return any(
        _salary_ranges_overlap(expected, advertised)
        for expected in expected_ranges
    )


def _salary_ranges_overlap(left: SalaryRange, right: SalaryRange) -> bool:
    """判断两个开放区间是否有实际范围交集，不把相邻端点当作交集。"""
    lower_values = [
        value for value in (left.minimum, right.minimum)
        if value is not None
    ]
    upper_values = [
        value for value in (left.maximum, right.maximum)
        if value is not None
    ]
    lower = max(lower_values) if lower_values else None
    upper = min(upper_values) if upper_values else None
    if lower is None or upper is None:
        return True
    if lower < upper:
        return True
    return (
        lower == upper
        and left.minimum == left.maximum == lower
        and right.minimum == right.maximum == upper
    )


def match_jobs(documents, query, *, threshold=0.0, batch_size=16):
    """兼容纯文本文档和预先构造的职位 chunk 分组。"""
    if documents and isinstance(documents[0], str):
        return _match_text_jobs(
            documents,
            query,
            threshold=threshold,
            batch_size=batch_size,
        )
    return _match_job_chunks(
        documents,
        query,
        threshold=threshold,
        batch_size=batch_size,
    )


def deduplicate(values):
    """保持原顺序去重，并忽略空字符串。"""
    if isinstance(values, str):
        values = [values]
    result = []
    seen = set()
    for value in values or []:
        value = str(value or "").strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def card_matches_exclusions(card, excluded_keywords):
    """判断职位卡片是否包含用户明确排除的工作内容。"""
    searchable = clean_text(" ".join([
        str(card.get("jobName") or ""),
        str(card.get("postDescription") or ""),
        str(card.get("brandIndustry") or ""),
    ])).casefold()
    for raw_keyword in excluded_keywords:
        keyword = str(raw_keyword or "").strip()
        if keyword and keyword.casefold() in searchable:
            return True
    return False


def card_matches_search_scope(card, search_params=None):
    """按当前搜索地点限制候选卡片，保证 Qdrant 与本地回退一致。"""
    if search_params is None:
        return True
    locations = {
        str(value).strip().rstrip("市")
        for value in (getattr(search_params, "location", None) or [])
        if str(value).strip()
    }
    if (
        not locations
        or "全国" in locations
        or getattr(search_params, "location_unlimited", False)
    ):
        return True
    city = job_city(card)
    if not city:
        return True
    return city in locations


def _active_time_days(value):
    """把详情页的活跃文本转换为保守的“距现在天数”。未知格式返回 None。"""
    text = clean_text(value).casefold()
    if not text:
        return None
    if any(marker in text for marker in (
        "online",
        "在线",
        "刚刚",
        "刚才",
        "今天",
        "今日",
        "today",
        "hour",
        "minute",
        "小时",
        "分钟",
    )):
        return 0
    if "昨天" in text or "yesterday" in text:
        return 1
    match = re.search(r"(\d+)\s*(?:天|日)\s*(?:前|内)?", text)
    if match:
        return int(match.group(1))
    if any(marker in text for marker in ("本周", "这周", "近一周", "一周内", "this week")):
        return 7
    if any(marker in text for marker in ("本月", "这月", "近一月", "一个月内", "this month")):
        return 30
    return None


def _is_online_value(value) -> bool:
    """兼容职位列表接口返回的布尔值、数字和字符串状态。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value == 1
    return str(value or "").strip().casefold() in {"true", "1", "yes"}


def card_matches_boss_activity(
    card,
    search_params=None,
    *,
    allow_unknown=False,
):
    """按意图筛选 BOSS 活跃度，支持当前在线和最近 N 天活跃。"""
    activity = getattr(search_params, "boss_activity", "any") if search_params else "any"
    if activity == "any":
        return True

    if activity == "online":
        return _is_online_value(card.get("bossOnline"))

    if activity == "recent":
        days = getattr(search_params, "boss_activity_days", None)
        if not days:
            return False
        observed_days = _active_time_days(card.get("bossActiveTime"))
        if observed_days is None and _is_online_value(card.get("bossOnline")):
            observed_days = 0
        if observed_days is None:
            return bool(allow_unknown)
        return observed_days <= days

    return False


def card_passes_hard_filters(
    card,
    excluded_keywords,
    search_params=None,
    *,
    allow_unknown_activity=False,
    allow_unknown_salary=False,
):
    """执行不依赖语义模型的硬性过滤。"""
    return (
        not card_matches_exclusions(card, excluded_keywords or [])
        and card_matches_search_scope(card, search_params)
        and card_matches_salary(
            card,
            search_params,
            allow_unknown=allow_unknown_salary,
        )
        and card_matches_boss_activity(
            card,
            search_params,
            allow_unknown=allow_unknown_activity,
        )
    )


def match_card_batch(
    cards,
    query,
    excluded_keywords,
    search_params=None,
    *,
    threshold=default_match_threshold,
):
    """过滤一批职位并返回带相似度的匹配结果。"""
    excluded_keywords = excluded_keywords or []
    eligible_cards = [
        card for card in cards
        if card_passes_hard_filters(card, excluded_keywords, search_params)
    ]
    if not eligible_cards:
        return []
    from analysis_work_content import build_job_chunk_records
    job_chunks = [build_job_chunk_records(card) for card in eligible_cards]
    from vector_store import VectorStoreUnavailable, vector_store

    if not vector_store.enabled:
        return [
            {"job_card": eligible_cards[index], "score": score}
            for index, score in match_jobs(
                job_chunks,
                query,
                threshold=threshold,
            )
        ]

    try:
        ranked = vector_store.match_cards(
            eligible_cards,
            query,
            search_params=search_params,
            threshold=threshold,
        )
    except VectorStoreUnavailable as exc:
        from logging_config import get_logger

        get_logger(__name__).warning(
            "Qdrant 不可用，回退本地向量匹配: error=%s", exc
        )
        ranked = match_jobs(
            job_chunks,
            query,
            threshold=threshold,
        )
    return [
        {"job_card": eligible_cards[index], "score": score}
        for index, score in ranked
    ]
