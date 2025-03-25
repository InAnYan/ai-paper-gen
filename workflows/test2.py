import asyncio
import pickle
from pprint import pprint

from llama_index.core.workflow import Checkpoint, Context, Event, StartEvent, StopEvent, Workflow, WorkflowCheckpointer, step


class MiddleEvent(Event):
    content: str


class TestWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> MiddleEvent:
        return MiddleEvent(content="foo")

    @step
    async def finish(self, ev: MiddleEvent) -> StopEvent:
        return StopEvent(content=ev.content + " bar")


async def main():
    w = TestWorkflow(verbose=True)
    w = WorkflowCheckpointer(workflow=w)

    print("First run:")
    await w.run()

    checkpoints = list(w.checkpoints.items())[0][1]
    first_check = checkpoints[0]
    saved_first_check = first_check

    first_check = Checkpoint.model_validate(first_check)

    # print(saved_first_check.model_dump().items() ^ first_check.model_dump().items())
    pprint(saved_first_check)
    pprint(first_check)


    print("From checkpoint:")
    await w.run_from(checkpoint=first_check)


asyncio.run(main())
