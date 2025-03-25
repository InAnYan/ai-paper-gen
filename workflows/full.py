from typing import Optional
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step, Event

from workflows.indexer import Document, Index, IndexGeneratorWorkflow, IndexRequest
from workflows.paper_writer import PaperWriteRequest, PaperWriterWorkflow, WrittenPaper
from workflows.planner import Plan, PlannerRequest, PlannerWorkflow
from workflows.related_work_finder import RelatedWork, RelatedWorkFindRequest, RelatedWorkFinderWorkflow
from workflows.paper_translator import PaperTranslated, PaperTranslatorWorkflow, TranslatePaperRequest


class WritePaperRequest(StartEvent):
    title: str
    language: str
    translate_to_language: Optional[str]


class FinishedPaper(StopEvent):
    paper: WrittenPaper
    translated: Optional[PaperTranslated]
    references: RelatedWork


class NestedPlan(Event):
    plan: Plan


class NestedRelatedWork(Event):
    related_work: RelatedWork


class NestedIndex(Event):
    index: Index


class NestedWrittenPaper(Event):
    written_paper: WrittenPaper


class NestedPaperTranslated(Event):
    paper_translated: PaperTranslated
    

class FullWritePaperWorkflow(Workflow):
    papers_per_query: int

    def __init__(self, papers_per_query: int, **kwargs):
        super().__init__(**kwargs)
        self.papers_per_query = papers_per_query
    
    @step
    async def plan_paper(self, ev: WritePaperRequest, planner_workflow: PlannerWorkflow) -> NestedPlan:
        start = PlannerRequest(title=ev.title)
        res = await planner_workflow.run(start_event=start)
        return NestedPlan(plan=res)

    @step
    async def find_related_work(self, ev: NestedPlan, related_work_finder: RelatedWorkFinderWorkflow) -> NestedRelatedWork:
        start = RelatedWorkFindRequest(
            queries=ev.plan.search_queries,
            papers_per_query=self.papers_per_query,
        )
        
        res = await related_work_finder.run(start_event=start)
        
        return NestedRelatedWork(related_work=res)

    @step
    async def index_related_work(self, ev: NestedRelatedWork, index_generator: IndexGeneratorWorkflow) -> NestedIndex:
        start = IndexRequest(
            documents=[Document(id=str(id), content=ref.content) for id, ref in enumerate(ev.related_work.references)]
        )

        res = await index_generator.run(start_event=start)
        
        return NestedIndex(index=res)

    @step
    async def write_paper(self, ctx: Context, evs: WritePaperRequest | NestedPlan | NestedIndex, paper_writer: PaperWriterWorkflow) -> Optional[NestedWrittenPaper]:
        got: Optional[Tuple[WritePaperRequest, NestedPlan, NestedIndex]] = ctx.collect_events(evs, [WritePaperRequest, NestedPlan, NestedIndex])  # type: ignore
        
        if not got:
            return None
        
        req, plan, index = got

        start = PaperWriteRequest(
            title=req.title,
            plan=plan.plan,
            index=index.index,
            language=req.language,
        )
        
        res = await paper_writer.run(start_event=start)

        return NestedWrittenPaper(written_paper=res)

    @step
    async def translate_paper(self, ctx: Context, evs: WritePaperRequest | NestedWrittenPaper, translator: PaperTranslatorWorkflow) -> Optional[NestedPaperTranslated]:
        got: Optional[Tuple[WritePaperRequest, NestedWrittenPaper]] = ctx.collect_events(evs, [WritePaperRequest, NestedWrittenPaper])  # type: ignore
        
        if not got:
            return None
        
        req, paper = got

        if not req.translate_to_language:
            return None
        
        start = TranslatePaperRequest(
            paper=paper.written_paper,
            from_language=req.language,
            to_language=req.translate_to_language,
        )

        res = await translator.run(start_event=start)

        return NestedPaperTranslated(paper_translated=res)

    @step
    async def finish(self, ctx: Context, evs: NestedPaperTranslated | WritePaperRequest | NestedWrittenPaper | NestedRelatedWork) -> Optional[FinishedPaper]:
        got: Optional[Tuple[WritePaperRequest, NestedWrittenPaper, NestedRelatedWork]] = ctx.collect_events(evs, [WritePaperRequest, NestedWrittenPaper, NestedRelatedWork])  # type: ignore
        
        if not got:
            return None
        
        req, paper, references = got

        if req.translate_to_language:
            translated = await ctx.wait_for_event(PaperTranslated)
        else:
            translated = None

        return FinishedPaper(
            paper=paper.written_paper,
            references=references.related_work.references,
            translated=translated.paper_translated if translated else None,
        )
