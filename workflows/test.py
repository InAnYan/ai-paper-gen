import asyncio
import pickle
from pprint import pprint

from llama_index.core.workflow import Checkpoint, Context, Event, JsonSerializer, StartEvent, StopEvent, Workflow, WorkflowCheckpointer, step


class EventA(Event):
    content: str


class EventB(Event):
    content: str


class TestWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> EventA:
        print("RUN: start")
        return EventA(content="first")

    @step
    async def process(self, ev: EventA) -> EventB:
        print("RUN: process")
        return EventB(content="second: " + ev.content)

    @step
    async def finish(self, ctx: Context, ev: EventB) -> StopEvent:
        print("RUN: finish")
        return StopEvent(result="end: " + ev.content)
        

async def main():
    w = WorkflowCheckpointer(workflow=TestWorkflow())
    
    await w.run()
    
    checks = list(w.checkpoints.items())[0][1]
    check = checks[1]

    check = pickle.dumps(check)
    check = pickle.loads(check)

    print('-' * 40)
    
    await w.run_from(checkpoint=check)


async def main_old():
    w = TestWorkflow()
    ctx = Context(w)

    try:
        pprint(await w.run(ctx=ctx))
    except:
        print("ERROR")

    ctx = ctx.to_dict(serializer=JsonSerializer())
    ctx = Context.from_dict(w, ctx, serializer=JsonSerializer())
    await ctx.set('var', '1')
    
    print('-' * 40)
    pprint(await w.run(ctx=ctx))


asyncio.run(main())
