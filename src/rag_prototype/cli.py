"""Command-line interface for RAG prototype."""

import click
import json
from pathlib import Path
from typing import List

from .rag_pipeline import RAGPipeline
from .config import config


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """RAG Prototype - A simple Retrieval-Augmented Generation system."""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('sources', nargs=-1, required=True)
@click.option('--openai-embeddings/--local-embeddings', default=True, 
              help='Use OpenAI embeddings (default) or local embeddings')
def ingest(sources, openai_embeddings):
    """Ingest documents into the RAG system."""
    click.echo("Starting document ingestion...")
    
    # Validate sources
    valid_sources = []
    for source in sources:
        source_path = Path(source)
        if source_path.exists():
            valid_sources.append(str(source_path))
        else:
            click.echo(f"Warning: Source not found: {source}", err=True)
    
    if not valid_sources:
        click.echo("Error: No valid sources provided", err=True)
        return
    
    # Initialize RAG pipeline
    rag = RAGPipeline(use_openai_embeddings=openai_embeddings)
    
    # Ingest documents
    result = rag.ingest_documents(valid_sources)
    
    if result["success"]:
        click.echo(f"✅ {result['message']}")
        click.echo(f"📊 Processed {result['document_count']} document chunks")
    else:
        click.echo(f"❌ {result['message']}", err=True)


@cli.command()
@click.argument('text')
@click.option('--source-name', default='command_line', help='Name for the text source')
@click.option('--openai-embeddings/--local-embeddings', default=True,
              help='Use OpenAI embeddings (default) or local embeddings')
def ingest_text(text, source_name, openai_embeddings):
    """Ingest raw text into the RAG system."""
    click.echo("Ingesting text...")
    
    # Initialize RAG pipeline
    rag = RAGPipeline(use_openai_embeddings=openai_embeddings)
    
    # Ingest text
    metadata = {"source": source_name}
    result = rag.ingest_text(text, metadata=metadata)
    
    if result["success"]:
        click.echo(f"✅ {result['message']}")
        click.echo(f"📊 Created {result['document_count']} text chunks")
    else:
        click.echo(f"❌ {result['message']}", err=True)


@cli.command()
@click.argument('question')
@click.option('--k', default=4, help='Number of relevant documents to retrieve')
@click.option('--no-sources', is_flag=True, help='Do not show source documents')
@click.option('--openai-embeddings/--local-embeddings', default=True,
              help='Use OpenAI embeddings (default) or local embeddings')
@click.option('--json-output', is_flag=True, help='Output in JSON format')
def query(question, k, no_sources, openai_embeddings, json_output):
    """Ask a question to the RAG system."""
    # Initialize RAG pipeline
    rag = RAGPipeline(use_openai_embeddings=openai_embeddings)
    
    # Process query
    result = rag.query(question, k=k, return_sources=not no_sources)
    
    if json_output:
        click.echo(json.dumps(result, indent=2))
        return
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("QUESTION:")
    click.echo(question)
    click.echo("\n" + "="*60)
    click.echo("ANSWER:")
    click.echo(result["answer"])
    
    if not no_sources and result.get("sources"):
        click.echo("\n" + "="*60)
        click.echo(f"SOURCES ({result['source_count']} documents):")
        for i, source in enumerate(result["sources"], 1):
            click.echo(f"\n[{i}] {source['content']}")
            if source.get("metadata"):
                click.echo(f"    Metadata: {source['metadata']}")


@cli.command()
@click.option('--openai-embeddings/--local-embeddings', default=True,
              help='Use OpenAI embeddings (default) or local embeddings')
def interactive(openai_embeddings):
    """Start an interactive Q&A session."""
    click.echo("🤖 Starting interactive RAG session...")
    click.echo("Type 'quit' or 'exit' to end the session.")
    click.echo("Type 'info' to see system information.")
    click.echo("-" * 60)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(use_openai_embeddings=openai_embeddings)
    
    while True:
        try:
            question = click.prompt("\n❓ Your question", type=str)
            
            if question.lower() in ['quit', 'exit']:
                click.echo("👋 Goodbye!")
                break
            
            if question.lower() == 'info':
                info = rag.get_system_info()
                click.echo("\n📊 System Information:")
                click.echo(json.dumps(info, indent=2))
                continue
            
            # Process query
            result = rag.query(question, return_sources=True)
            
            click.echo("\n" + "="*60)
            click.echo("🤖 ANSWER:")
            click.echo(result["answer"])
            
            if result.get("sources"):
                click.echo(f"\n📚 Sources ({result['source_count']} documents):")
                for i, source in enumerate(result["sources"], 1):
                    content = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                    click.echo(f"  [{i}] {content}")
            
        except (KeyboardInterrupt, EOFError):
            click.echo("\n👋 Goodbye!")
            break
        except Exception as e:
            click.echo(f"❌ Error: {str(e)}", err=True)


@cli.command()
@click.option('--openai-embeddings/--local-embeddings', default=True,
              help='Use OpenAI embeddings (default) or local embeddings')
def info(openai_embeddings):
    """Show system information."""
    # Initialize RAG pipeline
    rag = RAGPipeline(use_openai_embeddings=openai_embeddings)
    
    # Get system info
    system_info = rag.get_system_info()
    
    click.echo("📊 RAG System Information:")
    click.echo(json.dumps(system_info, indent=2))


@cli.command()
def config_info():
    """Show current configuration."""
    click.echo("⚙️ Configuration:")
    click.echo(f"  OpenAI API Key: {'✅ Set' if config.openai_api_key else '❌ Not set'}")
    click.echo(f"  Model: {config.model_name}")
    click.echo(f"  Embedding Model: {config.embedding_model}")
    click.echo(f"  Chunk Size: {config.chunk_size}")
    click.echo(f"  Chunk Overlap: {config.chunk_overlap}")
    click.echo(f"  Max Tokens: {config.max_tokens}")
    click.echo(f"  Temperature: {config.temperature}")
    click.echo(f"  DB Path: {config.chroma_db_path}")


if __name__ == '__main__':
    cli()