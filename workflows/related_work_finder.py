# from __future__ import annotations
import logging
import asyncio
from pprint import pprint
from tempfile import NamedTemporaryFile
from scholarly import scholarly
from typing import Any, Dict, List, Optional
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from semanticscholar import SemanticScholar
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from llama_index.core import SimpleDirectoryReader
from urllib.parse import urlparse, urljoin
import urllib.parse

from semanticscholar.PaginatedResults import PaginatedResults
from semanticscholar.Paper import Paper


class RelatedWorkFindRequest(StartEvent):
    queries: List[str]
    papers_per_query: int


class Reference(Event):
    metadata: Dict[str, Any]  # BibTeX.
    url: str
    content: str

    def __str__(self) -> str:
        return f"Reference(metadata={self.metadata}, url={self.url}, content[:100]={self.content[:100]})"

    def __repr__(self) -> str:
        return self.__str__()

    
class RelatedWork(StopEvent):
    references: List[Reference]
    

class FindPaper(Event):
    query: str
    count: int


class PaperMetadata(Event):
    metadata: Dict[str, Any]  # BibTeX.
    url: str

    
class RelatedWorkFinderWorkflow(Workflow):
    sch: SemanticScholar
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sch = SemanticScholar()
                
    @step
    async def start(self, ctx: Context, ev: RelatedWorkFindRequest) -> FindPaper:
        await ctx.set('paper_count', len(ev.queries) * ev.papers_per_query)
        
        for query in ev.queries:
            for _ in range(ev.papers_per_query):
                ctx.send_event(FindPaper(
                    query=query,
                    count=ev.papers_per_query,
                ))

        return None  # type: ignore

    @step
    async def fetch_metadata(self, ctx: Context, ev: FindPaper) -> Optional[PaperMetadata]:
        try:
            papers: PaginatedResults = self.sch.search_paper(query=ev.query, open_access_pdf=True, bulk=True, limit=ev.count)  # type: ignore

            await self.decrement_paper_count(ctx, ev.count - len(papers.items))

            for paper in papers.items:
                paper: Paper

                ctx.send_event(PaperMetadata(
                    metadata=paper.raw_data,
                    url=paper.openAccessPdf['url']
                ))
        except Exception as e:
            logging.exception(e)
            await self.decrement_paper_count(ctx, ev.count)
            return None

    @step
    async def download_and_parse_paper(self, ctx: Context, ev: PaperMetadata) -> Optional[Reference]:
        try:
            r = requests.get(ev.url, timeout=10)

            with NamedTemporaryFile(suffix='.pdf') as temp_file:
                temp_file.write(r.content)
                temp_file.flush()

                docs = await SimpleDirectoryReader(input_files=[temp_file.name]).aload_data()

                return Reference(
                    metadata=ev.metadata,
                    url=ev.url,
                    content='\n'.join(doc.text for doc in docs)
                )
        except Exception as e:
            logging.exception(e)
            logging.warning(f"Skipping one paper: {ev.url}")
            await self.decrement_paper_count(ctx, 1)

    @step
    async def finish(self, ctx: Context, ev: Reference) -> Optional[RelatedWork]:
        paper_count = await self.get_paper_count(ctx)
        if not paper_count:
            return None
  
        result: Optional[List[Reference]] = ctx.collect_events(ev, [Reference] * paper_count)  # type: ignore
        if result is None:
            return None

        return RelatedWork(references=result)

    async def get_paper_count(self, ctx: Context) -> Optional[int]:
        paper_count = await ctx.get('paper_count')
        if not paper_count:
            return None
        
        assert isinstance(paper_count, int)

        return paper_count

    async def decrement_paper_count(self, ctx: Context, amount: int):
        if amount <= 0:
            return
        
        paper_count = await self.get_paper_count(ctx)

        if not paper_count:
            return

        await ctx.set('paper_count', paper_count - amount)

    async def has_fetched(self, ctx: Context, url: str) -> bool:
        urls: List[str] = await ctx.get('urls', [])
        return url in urls

    async def add_url(self, ctx: Context, url: str):
        urls = await ctx.get('urls', [])
        urls.append(url)
        await ctx.set('urls', urls)

    def is_banned(self, url: str) -> bool:
        BANNED = ['elibrary', '.ru', 'books', 'sciencedirect', 'emerald.com', 'springer']

        for ban in BANNED:
            if ban in url:
                return True

        return False


# Taken from: <https://stackoverflow.com/a/59547139/10037342>.
def base_url(url, with_path=False):
    parsed = urllib.parse.urlparse(url)
    path   = '/'.join(parsed.path.split('/')[:-1]) if with_path else ''
    parsed = parsed._replace(path=path)
    parsed = parsed._replace(params='')
    parsed = parsed._replace(query='')
    parsed = parsed._replace(fragment='')
    return parsed.geturl()


if __name__ == '__main__':
    async def main():
        workflow = RelatedWorkFinder(verbose=True, timeout=None)
        request = RelatedWorkFindRequest(
            queries=['doping in sports', 'cloud computing', 'large language models'],
            papers_per_query=2
        )
        result = await workflow.run(start_event=request)
        pprint(result)

    asyncio.run(main())
