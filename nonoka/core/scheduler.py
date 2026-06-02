import asyncio
from typing import Any, Protocol, runtime_checkable

from nonoka.core.session import Session, SessionStatus, StepStatus
from nonoka.core.types import RunResult
from nonoka.core.context import RunContext
from nonoka.core.plan import Step

@runtime_checkable
class Scheduler(Protocol):
  async def run(self, session: Session, runner: Any, **kwargs) -> RunResult:
    ...
  async def resume(self, session: Session, runner: Any) -> RunResult:
    ...

class ConversationalScheduler:
  """
  ReAct loop, sequential execution between turns.
  Parallel tool calls inside a single turn.
  """
  async def run(self, session: Session, runner: Any, prompt: str = "") -> RunResult:
    session.status = SessionStatus.RUNNING
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())
    
    # TODO:This is a mocked logic to simulate a ReAct turn execution without an actual LLM setup
    # In reality, this would loop and call runner.llm.chat()
    try:
      if prompt:
         # Simulate thinking...
         pass
         
      session.status = SessionStatus.COMPLETED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=True, session=session)
    except Exception as e:
      session.status = SessionStatus.FAILED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=False, session=session, error=str(e))
      
  async def resume(self, session: Session, runner: Any) -> RunResult:
    # Just continue the loop
    return await self.run(session, runner)

class DAGScheduler:
  """
  Topological Sort + Parallel Layer Execution.
  """
  def __init__(self, max_concurrency: int = 10):
    self.max_concurrency = max_concurrency

  async def run_plan(self, session: Session, runner: Any) -> RunResult:
    session.status = SessionStatus.RUNNING
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())
    plan = session.current_plan
    
    if not plan:
      return RunResult(success=False, session=session, error="No plan provided")

    sem = asyncio.Semaphore(self.max_concurrency)

    async def execute_limited(step: Step):
      async with sem:
        return await self._execute_step(step, session, runner)

    try:
      layers = plan.topological_groups()
      for layer_step_ids in layers:
        results = await asyncio.gather(*[
          execute_limited(plan.get_step(sid)) for sid in layer_step_ids if plan.get_step(sid)
        ], return_exceptions=True)

        # Handle failures
        if any(isinstance(r, Exception) for r in results):
          # We might add retry logic here in a full implementation
          session.status = SessionStatus.FAILED
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          return RunResult(success=False, session=session, error="A step failed during execution.")

        # Checkpoint layer
        await runner.checkpoint_store.save_session(session.session_id, session.to_state())
        
      session.status = SessionStatus.COMPLETED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=True, session=session)
    except Exception as e:
      session.status = SessionStatus.FAILED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=False, session=session, error=str(e))

  async def _execute_step(self, step: Step, session: Session, runner: Any) -> Any:
    # Look for capability
    capability = next((t for t in session.agent.tools if t.name == step.tool), None)
    if not capability:
      raise ValueError(f"Tool {step.tool} not found in agent.")

    ctx = RunContext(session)

    # Sync RUNNING status to both checkpoint and in-memory session
    session.step_statuses[step.id] = StepStatus.RUNNING
    await runner.checkpoint_store.save_step_status(session.session_id, step.id, StepStatus.RUNNING)

    # Execute (mocking step retry policy here for simplicity)
    result = await capability.invoke(ctx, step.args)

    # Sync COMPLETED status and result to both checkpoint and in-memory session
    from nonoka.core.session import StepResult
    session.completed_steps[step.id] = StepResult(data=result)
    session.step_statuses[step.id] = StepStatus.COMPLETED
    await runner.checkpoint_store.save_step_result(session.session_id, step.id, result)

    return result

  async def run(self, session: Session, runner: Any) -> RunResult:
    return await self.run_plan(session, runner)

  async def resume(self, session: Session, runner: Any) -> RunResult:
    # DAGScheduler can figure out which layers are already completed based on session.completed_steps
    # and just execute the remaining ones. Since run_plan doesn't check completed_steps explicitly yet,
    # we can modify it or just assume it skips. For this stage, we'll delegate to run_plan.
    return await self.run_plan(session, runner)


class HybridScheduler:
  """
  Smart switching between Conversational and DAG scheduler.
  """
  async def run(self, session: Session, runner: Any) -> RunResult:
    if not session.current_plan:
      session.current_plan = await runner._generate_plan(session, "auto")
    
    groups = session.current_plan.topological_groups()
    if len(groups) <= 1:
       return await ConversationalScheduler().run(session, runner)
    
    return await DAGScheduler().run_plan(session, runner)
