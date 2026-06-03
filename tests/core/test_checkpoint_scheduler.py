import asyncio
from nonoka.core.agent import Agent
from nonoka.core.runner import Runner
from nonoka.core.context import RunContext
from nonoka.core.plan import Plan, Step
from nonoka.backends.checkpoint.memory import MemoryCheckpointStore


async def test():
  from nonoka.core.tool import tool

  @tool
  async def add(ctx: RunContext, a: int, b: int) -> int:
    return a + b

  @tool
  async def multiply(ctx: RunContext, a: int, b: int) -> int:
    return a * b

  agent = Agent(model="test-model", tools=[add, multiply])

  # ------------------------------------------------------------------ #
  # 1. Test with default memory checkpoint (string shorthand)
  # ------------------------------------------------------------------ #
  runner = Runner(checkpoint="memory")

  plan = Plan(
    objective="Calculate something",
    steps=(
      Step(id="step1", tool="add", args={"a": 2, "b": 3}),
      Step(id="step2", tool="multiply", args={"a": 4, "b": 5}, depends_on=frozenset(["step1"])),
    )
  )

  result = await runner.run_plan(agent, plan=plan, deps=None)
  print(f"Run Result Success: {result.success}")
  if result.session:
    print(f"Final Status: {result.session.status}")

    state = await runner.checkpoint_store.load_session(result.session.session_id)
    if state:
      print(f"Checkpoint State Status: {state.status}")
      print(f"Checkpoint Step1 Result: {state.completed_steps['step1'].data}")
      print(f"Checkpoint Step2 Result: {state.completed_steps['step2'].data}")
      print(f"Checkpoint Step Statuses: {state.step_statuses}")

  # ------------------------------------------------------------------ #
  # 2. Test with explicit CheckpointStore instance (advanced usage)
  # ------------------------------------------------------------------ #
  custom_store = MemoryCheckpointStore()
  runner2 = Runner(checkpoint=custom_store)

  plan2 = Plan(
    objective="Another calculation",
    steps=(
      Step(id="s1", tool="add", args={"a": 10, "b": 20}),
    )
  )

  result2 = await runner2.run_plan(agent, plan=plan2, deps=None)
  print(f"\nCustom Store Result Success: {result2.success}")
  if result2.session:
    state2 = await custom_store.load_session(result2.session.session_id)
    if state2:
      print(f"Custom Store Step Statuses: {state2.step_statuses}")


if __name__ == "__main__":
  asyncio.run(test())