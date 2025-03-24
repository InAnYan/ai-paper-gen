from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLM, MessageRole
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field
import asyncio
from pprint import pprint

from util import ChatMessageTemplate, ChatTemplate


class PlannerRequest(StartEvent):
    title: str


@dataclass
class PlannedParagraph:
    what_will_be_written: str


@dataclass
class PlannedSection:
    name: str
    what_will_be_written: str
    paragraphs: List[PlannedParagraph]

class Plan(StopEvent):
    structure: List[PlannedSection]
    search_queries: List[str]


class PlannerWorkflow(Workflow):
    llm: LLM
    chat_template: ChatTemplate

    def __init__(self, llm: LLM, chat_template: ChatTemplate, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm.as_structured_llm(PlannerWorkflow.InternalPlan)
        self.chat_template = chat_template

    class InternalParagraph(BaseModel):
        """Paragraph in the section."""
        
        what_will_be_written: str = Field(description="What will be written about in this paragraph. Descrive very shortly.")

    class InternalSection(BaseModel):
        """Logical section in the paper."""
            
        name: str = Field(description="Name of the section.")
        what_will_be_written: str = Field(description="What will be written in this section. Shortly.")
        search_queries: List[str] = Field(description="What to search in the literature.")
        paragraphs: List['PlannerWorkflow.InternalParagraph'] = Field(
            description="List of paragraphs in the section.",
            min_length=2,
            max_length=3
        )

    class InternalPlan(BaseModel):
        """Plan (draft) of the paper."""
        
        sections: List['PlannerWorkflow.InternalSection'] = Field(description="List of the logical sections in the paper")

    @step
    async def plan_paper(self, ev: PlannerRequest) -> Plan:
        messages = self.chat_template.render(paper_title=ev.title)

        result = await self.llm.achat(messages)
        plan: InternalPlan = result.raw  # type: ignore

        return Plan(
            structure=[
                PlannedSection(
                    name=section.name,
                    what_will_be_written=section.what_will_be_written,
                    paragraphs=[
                        PlannedParagraph(paragraph.what_will_be_written)
                        for paragraph in section.paragraphs
                    ]
                )
                for section in plan.sections
            ],
            search_queries=[
                query
                for section in plan.sections
                for query in section.search_queries
            ],
        )


if __name__ == '__main__':
    async def main():
        templates = Environment(loader=FileSystemLoader('templates'))
        
        template = ChatTemplate(
            messages=[
                ChatMessageTemplate(
                    role=MessageRole.SYSTEM,
                    template=templates.get_template('plan_system.jinja'),
                ),
                ChatMessageTemplate(
                    role=MessageRole.USER,
                    template=templates.get_template('plan_user.jinja'),
                ),
            ]
        )
        
        workflow = PlannerWorkflow(llm=OpenAI(model='gpt-4o-mini'), chat_template=template, timeout=None, verbose=True)
        req = PlannerRequest(title='large language models')
        result = await workflow.run(start_event=req)
        pprint(result)
    
    asyncio.run(main())
