from typing import Optional
import click
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


@click.group()
def app():
    pass


@app.command()
@click.argument('title', type=str, help='Title of the paper to generate')
@click.argument('main_language', type=str, help='Main language of the paper')
@click.argument('output', type=click.File('w'), help='Output file', required=False)
@click.option('--template', type=click.File('r'), help='Template file', required=False)
@click.option('--additional_language', type=str, help='Translate parts of paper to other language additionally')
@click.option('--additional_metadata', type=str, help='Additional metadata (path to YAML file) to pass to the final environment')
@click.option('--llm_model', type=str, default='gpt-4o-mini', help='LLM model')
@click.option('--llm_provider', type=str, default='openai', help='LLM model provider (OpenAI, Google, etc.)')
@click.option('--embedding_model', type=str, help='Embedding model')
def generate_paper(
        title: str,
        main_language: str,
        output: Optional[click.File],
        template: Optional[click.File],
        additional_language: str,
        additional_metadata: str,
        llm_model: str,
        llm_provider: str,
        embedding_model: str
):
    """Generate paper using the title.

    If no template specified, output will be the context for template engine in JSON format.

    If no output file specified, program will output the results to `stdout`."""
    
    pass

@app.command()
@click.argument('output', type=click.Path, help='Path to image output')
def show_workflow_map(output: click.Path):
    """Visualize the internal workflow in an image."""

    pass


if __name__ == '__main__':
    app()
