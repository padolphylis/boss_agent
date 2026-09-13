"""筛选条件与城市编码。

数据存放在 data/ 目录下的 JSON 里，加载、缓存和查表全部在 CodeBook 内部完成，
外部只通过它的方法取值，拿不到原始字典。
"""

import json
from pathlib import Path


class CodeBook:
    """筛选条件与城市编码的统一入口。

    懒加载：首次取值时才读盘，之后复用内存里的字典。要重新加载就新建实例。
    """

    _options_file = "filter_options.json"
    _city_file = "city_codes.json"

    def __init__(self, data_dir: Path | str | None = None):
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self._options: dict[str, dict[str, int]] | None = None
        self._cities: dict[str, str] | None = None

    def code_of(self, category: str, label: str) -> int | str:
        """按维度取编码；标签为空或不存在时返回空串。"""
        return self._options_data().get(category, {}).get(label or "", "")

    def labels_of(self, category: str) -> list[str]:
        """取某个维度的全部标签，供提示词列举候选。"""
        return list(self._options_data().get(category, {}))

    def city_code(self, name: str) -> str:
        """城市名 -> 编码；识别不出返回空串，容忍「深圳市」这类后缀。"""
        normalized = (name or "").strip().rstrip("市")
        if not normalized:
            return ""
        for city, code in self._city_data().items():
            if city.rstrip("市") == normalized:
                return code
        return ""

    def city_names(self) -> list[str]:
        """取全部城市名，按长度倒序，便于优先命中更长的名字。"""
        return sorted(self._city_data(), key=len, reverse=True)

    def match_city(self, text: str, hint: str = "") -> str:
        """从文本里对齐出权威城市名，只做查表匹配，不做文本替换。

        hint 优先（例如 LLM 已给出的城市），其次在原文里找出现的城市名。
        """
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

    def _options_data(self) -> dict[str, dict[str, int]]:
        """加载筛选码表。"""
        if self._options is None:
            self._options = self._read_json(self._options_file)
        return self._options

    def _city_data(self) -> dict[str, str]:
        """加载城市表。"""
        if self._cities is None:
            self._cities = self._read_json(self._city_file)
        return self._cities

    def _read_json(self, filename: str) -> dict:
        return json.loads((self._data_dir / filename).read_text(encoding="utf-8"))


# 进程内共享的默认实例，避免各处重复读盘。
code_book = CodeBook()
