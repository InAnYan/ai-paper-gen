from typing import Optional, Tuple
import pickle
import json
import click
import logging
import jinja2
from pathlib import Path
import sys
import os

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.workflow import Checkpoint, WorkflowCheckpointer
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from jinja2 import Environment, Template, FileSystemLoader

from workflows.planner import PlannerWorkflow
from workflows.indexer import IndexGeneratorWorkflow
from workflows.full import FullWritePaperWorkflow, WritePaperRequest
from workflows.paper_writer import PaperWriterWorkflow, WrittenPaper
from workflows.related_work_finder import RelatedWorkFinderWorkflow

from util import quick_template


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stderr))


# from phoenix.otel import register

# # configure the Phoenix tracer
# tracer_provider = register(
#   project_name="my-llm-app", # Default is 'default'
#   auto_instrument=True # See 'Trace all calls made to a library' below
# )
# tracer = tracer_provider.get_tracer(__name__)


# from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


import asyncio
from functools import wraps

# Taken from: https://github.com/pallets/click/issues/85#issuecomment-503464628.
# I'm just too lazy.
def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


PAPERS_PER_QUERY = 1


@click.group()
def app():
    pass


@app.command()
@click.argument('title', type=str)
@click.argument('language', type=str)
@click.argument('output', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--templates-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Directory for LLM message templates', required=True, default="templates/")
@click.option('--llm-model', type=str, default='gpt-4o-mini', help='LLM model (currently, only OpenAI models are supported)')
@click.option('--embedding-model', type=str, default='text-embedding-3-small', help='LLM model (currently, only OpenAI models are supported)')
@click.option('--checkpoints-file', type=click.Path(), default='checkpoint.pickle', help='Checkpoints file (will be overwritten).', required=False)
@coro
async def generate_paper(
    title: str,
    language: str,
    output: click.Path,
    templates_dir: click.Path,
    llm_model: str,
    embedding_model: str,
    checkpoints_file: Optional[click.Path],
):
    """Generate paper using the title.

    Outputs JSON context."""

    full_workflow = build_workflow(
        templates_dir,
        llm_model,
        embedding_model,
    )

    start = WritePaperRequest(
        title=title,
        language=language,
    )

    try:
        result = await full_workflow.run(start_event=start)
        
        with open(str(output), 'w') as fout:
            json.dump(result.model_dump(), fout, ensure_ascii=False)
    except:
        logging.exception("Got an error")

    if checkpoints_file:
        with open(str(checkpoints_file), 'wb') as fout:
            checkpoint = list(full_workflow.checkpoints.items())[0][1][-1]
            pickle.dump(checkpoint, fout)


@app.command()
@click.argument('checkpoints-file', type=click.Path(), default='checkpoint.pickle')
@click.argument('output', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--templates-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Directory for LLM message templates', required=True, default="templates/")
@click.option('--llm-model', type=str, default='gpt-4o-mini', help='LLM model (currently, only OpenAI models are supported)')
@click.option('--embedding-model', type=str, default='text-embedding-3-small', help='LLM model (currently, only OpenAI models are supported)')
@coro
async def continue_checkpoint(
    checkpoints_file: click.Path,
    output: click.Path,
    templates_dir: click.Path,
    llm_model: str,
    embedding_model: str,
):
    with open(str(checkpoints_file), 'rb') as fin:
        checkpoint: Checkpoint = pickle.load(fin)

    full_workflow = build_workflow(
        templates_dir,
        llm_model,
        embedding_model,
    )

    try:
        result = await full_workflow.run_from(checkpoint=checkpoint)
        
        with open(str(output), 'w') as fout:
            json.dump(result.model_dump(), fout, ensure_ascii=False)
    except:
        logging.exception("Got an error")

    checkpoint_items = list(full_workflow.checkpoints.items())

    if checkpoint_items:
        with open(str(checkpoints_file), 'wb') as fout:
            checkpoint = checkpoint_items[0][1][-1]
            pickle.dump(checkpoint, fout)


@app.command()
@click.argument('context', type=click.File('r'))
@click.argument('template', type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.argument('output_paper', type=click.File('w'))
@click.argument('output_bibliography', type=click.File('w'))
def fill_template(context: click.File, template: click.File, output_paper: click.File, output_bibliography: click.File):
    """Use Jinja templates to fill a context generated from `generate-paper`.

    `template` can be a directory of templates (root template must be `root.jinja`). It can be a file also."""

    context = json.loads(context.read())
    
    if os.path.isdir(str(template)):
        templates = Environment(loader=FileSystemLoader(str(template)))
        output_paper.write(templates.get_template('root.jinja').render(context))
    else:
        template = Template(template.read())
        output_paper.write(template.render(context))

    # Next code is highly shitty. But I'm too lazy to make a proper product.

    for id, reference in enumerate(context['references']['references']):
        bibtex = reference['metadata']['citationStyles']['bibtex']
        old_key = bibtex[bibtex.find('{')+1:bibtex.find(',')]
        bibtex = bibtex.replace(old_key, f'ref{id}')
        output_bibliography.write(bibtex + "\n\n")
    

def build_workflow(
    templates_dir: click.Path,
    llm_model: str,
    embedding_model: str,
) -> WorkflowCheckpointer:
    llm = OpenAI(model=llm_model)
    embeddings = OpenAIEmbedding(model=embedding_model)

    templates = jinja2.Environment(loader=jinja2.FileSystemLoader(str(templates_dir)))

    planner_workflow = PlannerWorkflow(
        llm=llm,
        chat_template=quick_template(templates, "plan"),
        verbose=True,
        timeout=None,
    )

    related_finder_workflow = RelatedWorkFinderWorkflow(
        verbose=True,
        timeout=None,
    )

    token_text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    
    indexer_workflow = IndexGeneratorWorkflow(
        splitter=token_text_splitter,
        embedding_model=embeddings,
        verbose=True,
        timeout=None,
    )

    paper_writer_workflow = PaperWriterWorkflow(
        llm=llm,
        embedding_model=embeddings,
        goal_template=quick_template(templates, "goal"),
        relevance_template=quick_template(templates, "relevance"),
        main_material_system_template=templates.get_template("write_system.jinja"),
        main_material_section_template=templates.get_template("write_section_user.jinja"),
        main_material_paragraph_template=templates.get_template("write_paragraph_user.jinja"),
        conclusions_template=quick_template(templates, "conclusions"),
        annotation_template=quick_template(templates, "annotation"),
        keywords_template=quick_template(templates, "keywords"),
        udc_template=quick_template(templates, "udc"),
        verbose=True,
        timeout=None,
    )

    full_workflow = FullWritePaperWorkflow(
        papers_per_query=PAPERS_PER_QUERY,
        verbose=True,
        timeout=None,
    )

    full_workflow.add_workflows(
        planner_workflow=planner_workflow,
        related_work_finder=related_finder_workflow,
        index_generator=indexer_workflow,
        paper_writer=paper_writer_workflow,
    )

    return WorkflowCheckpointer(workflow=full_workflow)


@app.command()
@click.argument('output-dir', type=click.Path(dir_okay=True, exists=True, file_okay=True))
def show_workflow_map(output_dir: click.Path):
    """Visualize the internal workflow in images."""

    draw_all_possible_flows(FullWritePaperWorkflow(None), filename=str(Path(str(output_dir)) / 'full_workflow.html'))  # type: ignore
    draw_all_possible_flows(PlannerWorkflow(None, None), filename=str(Path(str(output_dir)) / 'planner.html'))  # type: ignore
    draw_all_possible_flows(RelatedWorkFinderWorkflow(), filename=str(Path(str(output_dir)) / 'related_work.html'))  # type: ignore
    draw_all_possible_flows(IndexGeneratorWorkflow(None, None), filename=str(Path(str(output_dir)) / 'index.html'))  # type: ignore
    draw_all_possible_flows(PaperWriterWorkflow(None, None, None, None, None, None, None, None, None, None, None), filename=str(Path(str(output_dir)) / 'writer.html'))  # type: ignore
    draw_all_possible_flows(PaperTranslatorWorkflow(None, None), filename=str(Path(str(output_dir)) / 'translator.html'))  # type: ignore


if __name__ == '__main__':
    app()
