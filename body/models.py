from enum import Enum

from pydantic import BaseModel, Field


class Job(BaseModel):
    job_id: str
    title: str
    company: str
    city: str | None = None
    salary: str | None = None
    experience: str | None = None
    education: str | None = None
    description: str = ""
    source_url: str | None = None


class JobTarget(BaseModel):
    titles: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)
    min_salary: int | None = None
    required_keywords: list[str] = Field(default_factory=list)
    excluded_keywords: list[str] = Field(default_factory=list)


class PageObservation(BaseModel):
    url: str
    title: str = ""
    dom_text: str = ""
    ocr_text: str = ""
    screenshot_path: str | None = None
    page_type: str = "unknown"


class ActionType(str, Enum):
    READ_JOB = "read_job"
    SKIP_JOB = "skip_job"
    WAIT_FOR_USER = "wait_for_user"
    FILL_MESSAGE = "fill_message"
    SUBMIT_APPLICATION = "submit_application"


class AgentAction(BaseModel):
    action: ActionType
    reason: str
    job_id: str | None = None
    message: str | None = None


class JobAnalysis(BaseModel):
    recommended: bool
    score: int = Field(ge=0, le=100)
    matched_items: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    reason: str
    message: str


class JobState(str, Enum):
    NEW = "new"
    OBSERVED = "observed"
    ANALYZED = "analyzed"
    WAITING_CONFIRMATION = "waiting_confirmation"
    APPROVED = "approved"
    SUBMITTING = "submitting"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    BLOCKED = "blocked"
