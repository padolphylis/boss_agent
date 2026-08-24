from pathlib import Path

from playwright.sync_api import sync_playwright

from body.browser_observer import BrowserObserver


LOCAL_TEST_PAGE = Path("data/local-job.html")
EVIDENCE_DIR = "data/evidence"


def run() -> None:
    if not LOCAL_TEST_PAGE.exists():
        raise FileNotFoundError(
            f"找不到本地测试页面：{LOCAL_TEST_PAGE}。"
            "请先创建本地 HTML，不要直接访问真实招聘平台。"
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(LOCAL_TEST_PAGE.resolve().as_uri(), wait_until="domcontentloaded")

        observation = BrowserObserver(page, EVIDENCE_DIR).observe("local-job")
        if observation.page_type == "blocked":
            raise RuntimeError("检测到访问受限页面，任务已阻断。")

        print(f"页面标题: {observation.title}")
        print(f"页面地址: {observation.url}")
        print(f"页面类型: {observation.page_type}")
        print(f"截图路径: {observation.screenshot_path}")
        print("页面文本:")
        print(observation.dom_text[:2000])

        input("按 Enter 关闭本地测试浏览器：")
        browser.close()


if __name__ == "__main__":
    run()
