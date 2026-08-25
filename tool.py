from DrissionPage import ChromiumPage
from langchain_core.tools import tool

# @tool("open_browser")
def open_browser(page: ChromiumPage, query: str) -> str:
    """打开指定网页，并返回网页标题。

    Args:
        query: 要打开的网页 URL，必须是以 http:// 或 https:// 开头的完整地址。

    Returns:
        网页标题。如果网页没有标题，则返回空字符串。
    """
    page.get(query)
    with open("zhipin.html", "w", encoding="utf-8") as f:
        f.write(page.html)
    return page.html

if __name__ == "__main__":
    page = ChromiumPage()
    print(open_browser(page, "https://www.zhipin.com/"))