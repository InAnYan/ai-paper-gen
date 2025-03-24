import asyncio
from pprint import pprint
from dataclasses import dataclass
from typing import List, Optional


from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.core.node_parser import TextSplitter, TokenTextSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import llama_index.core
from pydantic import BaseModel


class Document(BaseModel):
    id: str
    content: str


class IndexRequest(StartEvent):
    documents: List[Document]


class DocumentToIndex(Event):
    document: Document


class Node(BaseModel):
    document_id: str
    content: str
    vector: List[float]


class Index(StopEvent):
    entries: List[Node]


class Nodes(Event):
    nodes: List[Node]
    

class IndexGeneratorWorkflow(Workflow):
    splitter: TextSplitter
    embedding_model: BaseEmbedding

    def __init__(self, splitter: TextSplitter, embedding_model: BaseEmbedding, **kwargs):
        super().__init__(**kwargs)
        self.splitter = splitter
        self.embedding_model = embedding_model

    @step
    async def start(self, ctx: Context, ev: IndexRequest) -> DocumentToIndex:
        await ctx.set('count', len(ev.documents))
        for doc in ev.documents:
            ctx.send_event(DocumentToIndex(document=doc))

        return None  # type: ignore

    @step
    async def index_document(self, ev: DocumentToIndex) -> Nodes:
        return Nodes(
            nodes=[
                Node(
                    document_id=ev.document.id,
                    content=part.get_content(),
                    vector=await self.embedding_model.aget_text_embedding(part.get_content()),
                )
                for part in await self.splitter.aget_nodes_from_documents([llama_index.core.Document(text=ev.document.content)])
            ]
        )

    @step
    async def finish(self, ctx: Context, ev: Nodes) -> Optional[Index]:
        count = await ctx.get('count')
  
        result: Optional[List[Nodes]] = ctx.collect_events(ev, [Nodes] * count)  # type: ignore
        if result is None:
            return None

        return Index(
            entries=[
                node
                for nodes in result
                for node in nodes.nodes
            ],
        )


if __name__ == '__main__':
    async def main():
        workflow = IndexGeneratorWorkflow(splitter=TokenTextSplitter(), embedding_model=OpenAIEmbedding(), verbose=True, timeout=None)
        req = IndexRequest(
            documents=[
                Document(
                    id='doc1',
                    content='Hello',
                ),
                Document(
                    id='doc2',
                    content='Bye',
                ),
            ]
        )
        result = await workflow.run(start_event=req)
        pprint(result)

    asyncio.run(main())
