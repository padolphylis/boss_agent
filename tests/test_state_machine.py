import pytest

from body.models import ActionType, AgentAction, JobState
from body.state_machine import JobStateMachine


def test_job_flow_stops_for_confirmation() -> None:
    machine = JobStateMachine()
    machine.move_through(
        [
            JobState.OBSERVED,
            JobState.ANALYZED,
            JobState.WAITING_CONFIRMATION,
        ]
    )

    assert machine.state is JobState.WAITING_CONFIRMATION
    assert machine.can_move(JobState.APPROVED)
    assert not machine.can_move(JobState.SUBMITTING)


def test_invalid_transition_is_rejected() -> None:
    machine = JobStateMachine()

    with pytest.raises(ValueError):
        machine.move(JobState.SUBMITTING)


def test_recommended_action_waits_for_user() -> None:
    action = AgentAction(
        action=ActionType.WAIT_FOR_USER,
        job_id="job-1",
        reason="岗位和目标匹配",
        message="您好，我对该岗位很感兴趣。",
    )

    assert action.action is ActionType.WAIT_FOR_USER
    assert action.action is not ActionType.SUBMIT_APPLICATION
