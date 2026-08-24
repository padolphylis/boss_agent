from body.browser_observer import BrowserObserver


class FakeLocator:
    def __init__(self, text: str) -> None:
        self.text = text

    def inner_text(self) -> str:
        return self.text


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def locator(self, selector: str) -> FakeLocator:
        assert selector == "body"
        return FakeLocator(self.text)


def test_detect_login_page():
    observer = BrowserObserver(FakePage("请登录后查看职位"))

    assert observer._detect_page_type() == "login_or_verification"


def test_detect_job_page():
    observer = BrowserObserver(FakePage("Python 后端开发 招聘职位"))

    assert observer._detect_page_type() == "job_page"


def test_detect_blocked_page():
    observer = BrowserObserver(FakePage("访问受限，您的 IP 存在异常行为"))

    assert observer._detect_page_type() == "blocked"
