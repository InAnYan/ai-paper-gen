from typing import Optional
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

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


class FullWritePaperWorkflow(Workflow):
    papers_per_query: int

    def __init__(self, papers_per_query: int, **kwargs):
        super().__init__(**kwargs)
        self.papers_per_query = papers_per_query
    
    @step
    async def plan_paper(self, ev: WritePaperRequest, planner_workflow: PlannerWorkflow) -> Plan:
        start = PlannerRequest(title=ev.title)
        return await planner_workflow.run(start_event=start)

    @step
    async def find_related_work(self, ev: Plan, related_work_finder: RelatedWorkFinderWorkflow) -> RelatedWork:
        start = RelatedWorkFindRequest(
            queries=ev.search_queries,
            papers_per_query=self.papers_per_query,
        )
        return await related_work_finder.run(start_event=start)

    @step
    async def index_related_work(self, ev: RelatedWork, index_generator: IndexGeneratorWorkflow) -> Index:
        start = IndexRequest(
            documents=[Document(id=str(id), content=ref.content) for id, ref in enumerate(ev.references)]
        )
        return await index_generator.run(start_event=start)

    @step
    async def write_paper(self, ctx: Context, evs: WritePaperRequest | Plan | Index, paper_writer: PaperWriterWorkflow) -> Optional[WrittenPaper]:
        got: Optional[Tuple[WritePaperRequest, Plan, Index]] = ctx.collect_events(evs, [WritePaperRequest, Plan, Index])  # type: ignore
        
        if not got:
            return None
        
        req, plan, index = got

        start = PaperWriteRequest(
            title=req.title,
            plan=plan,
            index=index,
            language=req.language,
        )
        
        return await paper_writer.run(start_event=start)

    @step
    async def translate_paper(self, ctx: Context, evs: WritePaperRequest | WrittenPaper, translator: PaperTranslatorWorkflow) -> Optional[PaperTranslated]:
        got: Optional[Tuple[WritePaperRequest, WrittenPaper]] = ctx.collect_events(evs, [WritePaperRequest, WrittenPaper])  # type: ignore
        
        if not got:
            return None
        
        req, paper = got

        if not req.translate_to_language:
            return None
        
        start = TranslatePaperRequest(
            paper=paper,
            from_language=req.language,
            to_language=req.translate_to_language,
        )

        return await translator.run(start_event=start)

    @step
    async def finish(self, ctx: Context, evs: WritePaperRequest | WrittenPaper | RelatedWork) -> Optional[FinishedPaper]:
        got: Optional[Tuple[WritePaperRequest, WrittenPaper, RelatedWork]] = ctx.collect_events(evs, [WritePaperRequest, WrittenPaper, RelatedWork])  # type: ignore
        
        if not got:
            return None
        
        req, paper, references = got

        if req.translate_to_language:
            translated = await ctx.wait_for_event(PaperTranslated)
        else:
            translated = None

        return FinishedPaper(
            paper=paper,
            references=references,
            translated=translated,
        )


