# from __future__ import annotations
import asyncio
from pprint import pprint
from tempfile import NamedTemporaryFile
from scholarly import scholarly
from typing import Any, Dict, List, Optional
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from scholarly import scholarly
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from llama_index.core import SimpleDirectoryReader
from urllib.parse import urlparse, urljoin
import urllib.parse


class RelatedWorkFindRequest(StartEvent):
    queries: List[str]
    papers_per_query: int


class Reference(Event):
    metadata: Dict[str, Any]  # BibTeX or BibLaTeX...
    url: str
    content: str

    
class RelatedWork(StopEvent):
    references: List[Reference]
    

class FindPaper(Event):
    query: str


class PaperMetadata(Event):
    metadata: Dict[str, Any]  # BibTeX or BibLaTeX...
    url: str

    
class RelatedWorkFinderWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: RelatedWorkFindRequest) -> FindPaper:
        await ctx.set('paper_count', len(ev.queries) * ev.papers_per_query)
        
        for query in ev.queries:
            for _ in range(ev.papers_per_query):
                ctx.send_event(FindPaper(
                    query=query,
                ))

        return None  # type: ignore

    @step
    async def fetch_metadata(self, ctx: Context, ev: FindPaper) -> Optional[PaperMetadata]:
        it = scholarly.search_pubs(ev.query)

        try:
            pub = next(it)
            # assert 'pub_url' in pub
            url = pub['pub_url']

            while self.is_banned(url) or await self.has_fetched(ctx, url):
                pub = next(it)
                # assert 'pub_url' in pub
                url = pub['pub_url']

            # assert 'bib' in pub

            await self.add_url(ctx, pub['pub_url'])

            return PaperMetadata(
                metadata=pub['bib'],  # type: ignore
                url=pub['pub_url'],
            )

        except StopIteration:
            await ctx.set('paper_count', (self.get_paper_count(ctx) or 1) - 1)
            return None

    @step
    async def download_and_parse_paper(self, ev: PaperMetadata) -> Optional[Reference]:
        url = ev.url
        r = requests.get(url)

        if not str(urlparse(url).path).endswith('pdf'): # It doesn't work as I thought.
            soup = BeautifulSoup(r.text, 'html.parser')
        
            for link in soup.find_all('a', href=True):
                href = link['href']  # type: ignore
                href: str

                if str(urlparse(href).path).endswith('.pdf'):
                    r = requests.get(urljoin(base_url(url), href))
                    break

        with NamedTemporaryFile() as temp_file:
            temp_file.write(r.content)
            temp_file.flush()

            docs = await SimpleDirectoryReader(input_files=[temp_file.name]).aload_data()

            return Reference(
                metadata=ev.metadata,
                url=ev.url,
                content='\n'.join(doc.text for doc in docs)
            )

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

    async def has_fetched(self, ctx: Context, url: str) -> bool:
        urls: List[str] = await ctx.get('urls', [])
        return url in urls

    async def add_url(self, ctx: Context, url: str):
        urls = await ctx.get('urls', [])
        urls.append(url)
        await ctx.set('urls', urls)

    def is_banned(self, url: str) -> bool:
        BANNED = ['elibrary', '.ru', 'books', 'sciencedirect']

        for ban in BANNED:
            if url in ban:
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
