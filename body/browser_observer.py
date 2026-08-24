from pathlib import Path

from playwright.sync_api import Page

from .models import PageObservation


class BrowserObserver:
    """只读浏览器观察器，不执行登录、验证码或提交动作。"""

    def __init__(self, page: Page, evidence_dir: str = "data/evidence") -> None:
        self.page = page
        self.evidence_dir = Path(evidence_dir)

    def observe(self, evidence_name: str = "page") -> PageObservation:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = self.evidence_dir / f"{evidence_name}.png"
        self.page.screenshot(path=str(screenshot_path), full_page=True)
        return PageObservation(
            url=self.page.url,
            title=self.page.title(),
            dom_text=self.page.locator("body").inner_text(),
            screenshot_path=str(screenshot_path),
            page_type=self._detect_page_type(),
        )

    def _detect_page_type(self) -> str:
        text = self.page.locator("body").inner_text().lower()
        if any(
            keyword in text
            for keyword in ("访问受限", "异常行为", "access denied", "forbidden")
        ):
            return "blocked"
        if any(keyword in text for keyword in ("登录", "验证码", "login", "captcha")):
            return "login_or_verification"
        if any(keyword in text for keyword in ("职位", "job", "招聘")):
            return "job_page"
        return "unknown"
