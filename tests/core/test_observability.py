import pytest
from nonoka.core.event import AgentEvent, EventType
from nonoka.core.context import RunContext
from nonoka.core.session import Session
from nonoka.core.agent import Agent


@pytest.mark.asyncio
async def test_run_context_emit_structlog(capsys):
    """RunContext.emit() should log via structlog to stdout."""
    agent = Agent(model="test")
    session = Session(session_id="sess-1", agent=agent, deps=None)
    ctx = RunContext(session)

    event = AgentEvent(
        type=EventType.SESSION_STARTED,
        session_id="sess-1",
        data={"agent_id": "test-agent"}
    )
    ctx.emit(event)

    captured = capsys.readouterr()
    assert "session.started" in captured.out
    assert "sess-1" in captured.out


@pytest.mark.asyncio
async def test_run_context_checkpoint_request(capsys):
    """RunContext.checkpoint() should emit a checkpoint request event."""
    agent = Agent(model="test")
    session = Session(session_id="sess-2", agent=agent, deps=None)
    ctx = RunContext(session)

    await ctx.checkpoint(label="mid-flow")

    captured = capsys.readouterr()
    assert "checkpoint.requested" in captured.out
    assert "sess-2" in captured.out
