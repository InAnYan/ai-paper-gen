from typing import Optional, Tuple
import pickle
import json
import click
import logging
import jinja2
from pathlib import Path
import sys

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.workflow import Checkpoint, WorkflowCheckpointer
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from workflows.planner import PlannerWorkflow
from workflows.paper_translator import PaperTranslatorWorkflow
from workflows.indexer import IndexGeneratorWorkflow
from workflows.full import FullWritePaperWorkflow, WritePaperRequest
from workflows.paper_writer import PaperWriterWorkflow
from workflows.related_work_finder import RelatedWorkFinderWorkflow

from util import quick_template


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stderr))


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
@click.option('--translate-to-language', type=str, help='Translate parts of paper to other language additionally')
@click.option('--llm-model', type=str, default='gpt-4o-mini', help='LLM model (currently, only OpenAI models are supported)')
@click.option('--embedding-model', type=str, default='text-embedding-3-small', help='LLM model (currently, only OpenAI models are supported)')
@click.option('--checkpoints-file', type=click.Path(), default='checkpoint.pickle', help='Checkpoints file (will be overwritten).', required=False)
@coro
async def generate_paper(
    title: str,
    language: str,
    output: click.Path,
    templates_dir: click.Path,
    translate_to_language: str,
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
        translate_to_language=translate_to_language,
    )

    try:
        result = await full_workflow.run(start_event=start)
        
        with open(str(output), 'w') as fout:
            json.dump(result, fout)
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
    checkpoints_path: click.Path,
    output: click.Path,
    templates_dir: click.Path,
    llm_model: str,
    embedding_model: str,
):
    with open(str(checkpoints_path), 'rb') as fin:
        checkpoint: Checkpoint = pickle.load(fin)
    
    full_workflow = build_workflow(
        templates_dir,
        llm_model,
        embedding_model,
    )

    try:
        result = await full_workflow.run_from(checkpoint=checkpoint)
        
        with open(str(output), 'w') as fout:
            json.dump(result, fout)
    except:
        logging.exception("Got an error")

    with open(str(checkpoints_path), 'wb') as fout:
        checkpoint = list(full_workflow.checkpoints.items())[0][1][-1]
        pickle.dump(checkpoint, fout)


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
        timeout=1000,
    )

    related_finder_workflow = RelatedWorkFinderWorkflow(
        verbose=True,
        timeout=1000,
    )

    token_text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    
    indexer_workflow = IndexGeneratorWorkflow(
        splitter=token_text_splitter,
        embedding_model=embeddings,
        verbose=True,
        timeout=1000,
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
        timeout=1000,
    )

    full_workflow = FullWritePaperWorkflow(
        papers_per_query=PAPERS_PER_QUERY,
        verbose=True,
        timeout=1000,
    )

    paper_translator = PaperTranslatorWorkflow(
        llm=llm,
        translate_template=quick_template(templates, "translate"),
        verbose=True,
        timeout=1000,
    )

    full_workflow.add_workflows(
        planner_workflow=planner_workflow,
        related_work_finder=related_finder_workflow,
        index_generator=indexer_workflow,
        paper_writer=paper_writer_workflow,
        translator=paper_translator,
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
