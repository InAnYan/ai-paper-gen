from dataclasses import dataclass
from typing import List, Optional, Tuple
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step, Context

from util import ChatTemplate, quick_process
from workflows.indexer import Index, Node
from workflows.planner import Plan

import jinja2


class PaperWriteRequest(StartEvent):
    title: str
    plan: Plan
    index: Index
    language: str


class PaperTitle(Event):
    content: str


class PaperPlan(Event):
    plan: Plan


class PaperIndex(Event):
    index: Index

    
class PaperLanguage(Event):
    content: str
    

@dataclass
class Sentence:
    content: str
    cites: List[int]


@dataclass
class Paragraph:
    sentences: List[Sentence]


class MainMaterial(Event):
    paragraphs: List[Paragraph]


class WrittenPaper(StopEvent):
    udc: str
    annotation: str
    keywords: List[str]
    goal: str
    relevance: Paragraph
    main_material: MainMaterial
    conclusions: List[str]


class Goal(Event):
    content: str


class Relevance(Event):
    content: Paragraph
    

class Conclusions(Event):
    content: List[str]


class Annotation(Event):
    content: str


class Keywords(Event):
    content: List[str]


class Udc(Event):
    content: str


class MainMaterialRequest(Event):
    plan: Plan
    index: Index


class PaperWriterWorkflow(Workflow):
    llm: LLM
    embedding_model: BaseEmbedding
    goal_template: ChatTemplate
    relevance_template: ChatTemplate
    main_material_system_template: jinja2.Template
    main_material_section_template: jinja2.Template
    main_material_paragraph_template: jinja2.Template
    conclusions_template: ChatTemplate
    annotation_template: ChatTemplate
    keywords_template: ChatTemplate
    udc_template: ChatTemplate

    def __init__(
        self,
        llm: LLM,
        embedding_model: BaseEmbedding,
        goal_template: ChatTemplate,
        relevance_template: ChatTemplate,
        main_material_system_template: jinja2.Template,
        main_material_section_template: jinja2.Template,
        main_material_paragraph_template: jinja2.Template,
        conclusions_template: ChatTemplate,
        annotation_template: ChatTemplate,
        keywords_template: ChatTemplate,
        udc_template: ChatTemplate,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.llm = llm
        self.embedding_model = embedding_model
        self.goal_template = goal_template
        self.relevance_template = relevance_template
        self.main_material_system_template = main_material_system_template
        self.main_material_section_template = main_material_section_template
        self.main_material_paragraph_template = main_material_paragraph_template
        self.conclusions_template = conclusions_template
        self.annotation_template = annotation_template
        self.keywords_template = keywords_template
        self.udc_template = udc_template

    @step
    async def start(self, ctx: Context, ev: PaperWriteRequest) -> PaperLanguage | PaperTitle | PaperIndex | PaperPlan | None:
        ctx.send_event(PaperLanguage(content=ev.language))
        ctx.send_event(PaperTitle(content=ev.title))
        ctx.send_event(PaperIndex(index=ev.index))
        ctx.send_event(PaperPlan(plan=ev.plan))
        print("MADE: start")

    @step
    async def make_intermediate_1(self, ctx: Context, evs: PaperLanguage | PaperTitle) -> Optional[Goal]:
        got: Optional[Tuple[PaperLanguage, PaperTitle]] = ctx.collect_events(evs, [PaperLanguage, PaperTitle])  # type: ignore
        
        if not got:
            return None
        
        language, title = got

        print("MADE: make_goal")
        return Goal(content=await quick_process(
            self.llm,
            self.goal_template,
            language=language.content,
            title=title.content,
        ))

    @step
    async def make_intermediate_2(self, ctx: Context, evs: PaperTitle | PaperLanguage | Goal | PaperIndex) -> Optional[Relevance]:
        got: Optional[Tuple[PaperTitle, PaperLanguage, Goal, PaperIndex]] = ctx.collect_events(evs, [PaperTitle, PaperLanguage, Goal, PaperIndex])  # type: ignore
        
        if not got:
            return None
        
        title, language, goal, index = got

        print("MADE: make_relevance")
        return Relevance(
            content=parse_paragraph(
                await quick_process(
                    self.llm,
                    self.relevance_template,
                    title=title.content,
                    goal=goal.content,
                    language=language.content,
                    facts=await self.find_knowledge(index, 'relevance')
                )
            )
        )

    @step
    async def make_intermediate_3(self, ctx: Context, evs: PaperLanguage | PaperPlan | PaperIndex) -> Optional[MainMaterial]:
        got: Optional[Tuple[PaperLanguage, PaperPlan, PaperIndex]] = ctx.collect_events(evs, [PaperLanguage, PaperPlan, PaperIndex])  # type: ignore

        if not got:
            return None

        language, plan, index = got
        plan = plan.plan

        paragraphs: List[Paragraph] = []

        system_message = self.main_material_system_template.render(language=language.content)
    
        history = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_message,
            )
        ]

        for section in plan.structure:
            section_message = self.main_material_section_template.render(section=section)        
            history.append(ChatMessage(
                role=MessageRole.USER,
                content=section_message
            ))

            for paragraph in section.paragraphs:
                facts = await self.find_knowledge(index, paragraph.what_will_be_written, 3)
                paragraph_message = self.main_material_paragraph_template.render(
                    paragraph=paragraph,
                    facts=facts,
                )
                history.append(ChatMessage(
                    role=MessageRole.USER,
                    content=paragraph_message,
                ))

                res = await self.llm.achat(history)

                history.append(res.message)

                paragraphs.append(parse_paragraph(str(res.message.content)))

        print("MADE: make_main_material")
        return MainMaterial(paragraphs=paragraphs)

    @step
    async def make_intermediate_4(self, ctx: Context, evs: PaperTitle | PaperLanguage | Goal | MainMaterial) -> Optional[Conclusions]:
        got: Optional[Tuple[PaperTitle, PaperLanguage, Goal, MainMaterial]] = ctx.collect_events(evs, [PaperTitle, PaperLanguage, Goal, MainMaterial])  # type: ignore
        
        if not got:
            return None
        
        title, language, goal, main_material = got

        print("MADE: make_conclusions")
        return Conclusions(
            content=(await quick_process(
                self.llm,
                self.conclusions_template,
                language=language.content,
                title=title.content,
                goal=goal.content,
                content=main_material.paragraphs
            )).split('\n\n')
        )

    @step
    async def make_intermediate_5(self, ctx: Context, evs: PaperTitle | PaperLanguage | Goal | MainMaterial | Conclusions) -> Optional[Annotation]:
        got: Optional[Tuple[PaperTitle, PaperLanguage, Goal, MainMaterial, Conclusions]] = ctx.collect_events(evs, [PaperTitle, PaperLanguage, Goal, MainMaterial, Conclusions])  # type: ignore
        
        if not got:
            return None
        
        title, language, goal, main_material, conclusions = got

        print("MADE: make_annotation")
        return Annotation(
            content=await quick_process(
                self.llm,
                self.annotation_template,
                title=title.content,
                language=language.content,
                goal=goal.content,
                content=main_material,
                conclusions=conclusions.content,
            )
        )

    @step
    async def make_intermediate_6(self, ctx: Context, evs: PaperTitle | PaperLanguage | Goal | Annotation) -> Optional[Keywords]:
        got: Optional[Tuple[PaperTitle, PaperLanguage, Goal, Annotation]] = ctx.collect_events(evs, [PaperTitle, PaperLanguage, Goal, Annotation])  # type: ignore

        if not got:
            print("KEYWORDS: not enough")
            print("KEYWORDS: having " + str(ctx._events_buffer[ctx._get_full_path(type(evs))]))
            return None
        
        title, language, goal, annotation = got

        print("MADE: make_keywords")
        return Keywords(
            content=(await quick_process(
                self.llm,
                self.keywords_template,
                title=title.content,
                language=language.content,
                goal=goal.content,
                abstract=annotation.content,
            )).split(", ")
        )

    @step
    async def make_intermediate_7(self, ctx: Context, evs: PaperTitle | Goal | Annotation) -> Optional[Udc]:
        got: Optional[Tuple[PaperTitle, Goal, Annotation]] = ctx.collect_events(evs, [PaperTitle, Goal, Annotation])  # type: ignore
        
        if not got:
            return None

        title, goal, annotation = got

        print("MADE: make_udc")
        return Udc(
            content=await quick_process(
                self.llm,
                self.udc_template,
                title=title.content,
                goal=goal.content,
                abstract=annotation.content,
            )
        )

    @step
    async def finish(
        self,
        ctx: Context,
        evs: Udc | Annotation | Keywords | Goal | Relevance | MainMaterial | Conclusions
    ) -> Optional[WrittenPaper]:
        got: Optional[Tuple[Udc | Annotation | Keywords | Goal | Relevance | MainMaterial | Conclusions]] = ctx.collect_events(evs, [Udc, Annotation, Keywords, Goal, Relevance, MainMaterial, Conclusions])  # type: ignore

        if not got:
            return None

        udc, annotation, keywords, goal, relevance, main_material, conclusions = got

        return WrittenPaper(
            udc=udc.content,
            annotation=annotation.content,
            keywords=keywords.content,
            goal=goal.content,
            relevance=relevance.content,
            main_material=main_material,
            conclusions=conclusions.content,
        )

    async def find_knowledge(self, index: PaperIndex, query: str, count: int = 3) -> List[Node]:
        query_vector = await self.embedding_model.aget_text_embedding(query)

        facts = [
            (
                entry,
                sum(
                    v1 * v2
                    for v1, v2 in zip(query_vector, entry.vector)
                ),
            )
            for entry in index.index.entries
        ]
        
        facts.sort(key=lambda t: t[1], reverse=True)

        facts = facts[:count]

        for fact in facts:
            index.index.entries.remove(fact[0])

        return [t[0] for t in facts]


def parse_paragraph(text: str) -> Paragraph:
    sentences = []

    for sentence in text.split('.'):
        sentence = sentence.strip()
        
        if '[' in sentence:
            content, cites = sentence.split('[')
            content = content.strip()
            cites = cites.strip()
            cites = cites.removesuffix(']')
            cites = [cite.strip() for cite in cites.split(',')]
        else:
            content = sentence
            cites = []

        cites = [int(cite) for cite in cites]
        
        sentences.append(Sentence(
            content=content,
            cites=cites,
        ))

    return Paragraph(sentences=sentences)
