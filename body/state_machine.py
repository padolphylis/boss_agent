from collections.abc import Iterable

from .models import JobState


TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.NEW: {JobState.OBSERVED, JobState.FAILED},
    JobState.OBSERVED: {JobState.ANALYZED, JobState.FAILED},
    JobState.ANALYZED: {JobState.WAITING_CONFIRMATION, JobState.SKIPPED},
    JobState.WAITING_CONFIRMATION: {JobState.APPROVED, JobState.SKIPPED},
    JobState.APPROVED: {JobState.SUBMITTING, JobState.BLOCKED, JobState.FAILED},
    JobState.SUBMITTING: {JobState.SUCCESS, JobState.BLOCKED, JobState.FAILED},
    JobState.SUCCESS: set(),
    JobState.SKIPPED: set(),
    JobState.FAILED: set(),
    JobState.BLOCKED: set(),
}


class JobStateMachine:
    def __init__(self, initial: JobState = JobState.NEW) -> None:
        self.state = initial
        self.history: list[JobState] = [initial]

    def move(self, target: JobState) -> JobState:
        allowed = TRANSITIONS[self.state]
        if target not in allowed:
            raise ValueError(f"invalid transition: {self.state.value} -> {target.value}")
        self.state = target
        self.history.append(target)
        return self.state

    def can_move(self, target: JobState) -> bool:
        return target in TRANSITIONS[self.state]

    def move_through(self, states: Iterable[JobState]) -> JobState:
        for state in states:
            self.move(state)
        return self.state
