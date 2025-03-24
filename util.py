from dataclasses import dataclass
from typing import Any, List
from jinja2 import Environment, Template
from llama_index.core.llms import LLM, ChatMessage, MessageRole


@dataclass
class ChatMessageTemplate:
    role: MessageRole
    template: Template

    def render(self, **context: Any) -> ChatMessage:
        content = self.template.render(**context)
        return ChatMessage(role=self.role, content=content)
    

@dataclass
class ChatTemplate:
    messages: List[ChatMessageTemplate]

    def render(self, **context: Any) -> List[ChatMessage]:
        return [message.render(**context) for message in self.messages]


async def quick_process(llm: LLM, chat_template: ChatTemplate, **kwargs) -> str:
    return str((await llm.achat(chat_template.render(**kwargs))).message.content)


def quick_template(environment: Environment, template: str) -> ChatTemplate:
    return ChatTemplate([
        ChatMessageTemplate(
            role=MessageRole.SYSTEM,
            template=environment.get_template(f"{template}_system.jinja")
        ),
        ChatMessageTemplate(
            role=MessageRole.USER,
            template=environment.get_template(f"{template}_user.jinja")
        ),
    ])
