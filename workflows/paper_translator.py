from typing import List, Optional
from llama_index.core.llms import LLM
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step, Context
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy, RetryPolicy

from util import ChatTemplate, quick_process
from workflows.paper_writer import WrittenPaper


class TranslatePaperRequest(StartEvent):
    paper: WrittenPaper
    from_language: str
    to_language: str


class TranslatedAnnotation(Event):
    content: str


class TranslatedKeywords(Event):
    content: List[str]
    

class PaperTranslated(StopEvent):
    annotation: str
    keywords: List[str]


class PaperTranslatorWorkflow(Workflow):
    llm: LLM
    translate_template: ChatTemplate

    @step
    async def translate_annotation(self, ev: TranslatePaper) -> TranslatedAnnotation:
        return TranslatedAnnotation(
            content=await self.translate(ev, ev.paper.annotation)
        )

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))  # I'm too lazy to make a separate template for keywords. Though this would be better (need commas).
    async def translate_keyword(self, ev: TranslatePaper) -> TranslatedKeywords:
        return TranslatedKeywords(
            content=[keyword.strip() for keyword in (await self.translate(ev, ', '.join(ev.paper.keywords))).split(',')]
        )
        
    async def translate(self, ev: TranslatePaper, text: str) -> str:
        return await quick_process(
            self.llm,
            self.translate_template,
            from_language=ev.from_language,
            to_language=ev.to_language,
            text=text,
        )

    @step
    async def finish(self, ctx: Context, evs: TranslatedAnnotation | TranslatedKeywords) -> Optional[PaperTranslated]:
        got: Optional[Tuple[TranslatedAnnotation, TranslatedKeywords]] = ctx.collect_events(evs, [TranslatedAnnotation, TranslatedKeywords])  # type: ignore
        
        if not got:
            return None
        
        annotation, keywords = got

        return PaperTranslated(annotation=annotation, keywords=keywords)

