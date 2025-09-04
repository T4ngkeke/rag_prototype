#!/usr/bin/env python3
"""
Simple CLI for the RAG prototype.
Works without external dependencies.
"""

import sys
import json
from pathlib import Path

# Import our simple RAG implementation
from simple_rag import SimpleRAGPipeline


def print_help():
    """Print help information."""
    print("""
🤖 Simple RAG Prototype CLI

Usage:
  python simple_cli.py <command> [options]

Commands:
  ingest <file1> [file2] ...   - Ingest text files into the system
  ingest-text "<text>"         - Ingest raw text
  query "<question>"           - Ask a question
  interactive                  - Start interactive mode
  info                         - Show system information
  help                         - Show this help

Examples:
  python simple_cli.py ingest data/sample_ai_document.txt
  python simple_cli.py query "What is Python?"
  python simple_cli.py interactive
""")


def ingest_files(rag, file_paths):
    """Ingest files into the RAG system."""
    print(f"🔄 Ingesting {len(file_paths)} files...")
    
    # Validate files
    valid_files = []
    for file_path in file_paths:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    if not valid_files:
        print("❌ No valid files to ingest")
        return
    
    result = rag.ingest_documents(valid_files)
    
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['message']}")


def ingest_text(rag, text):
    """Ingest raw text."""
    print("🔄 Ingesting text...")
    
    result = rag.ingest_text(text, {"source": "command_line"})
    
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['message']}")


def query(rag, question):
    """Process a query."""
    print(f"🔍 Searching for: {question}")
    
    result = rag.query(question)
    
    print("\n" + "="*60)
    print("QUESTION:")
    print(question)
    
    print("\n" + "="*60)
    print("ANSWER:")
    print(result["answer"])
    
    if result.get("sources"):
        print("\n" + "="*60)
        print(f"SOURCES ({result['source_count']} documents):")
        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source['content']}")
            if source.get("metadata"):
                print(f"    Source: {source['metadata']}")


def interactive_mode(rag):
    """Start interactive mode."""
    print("🤖 Starting interactive RAG session...")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'info' to see system information.")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'info':
                show_info(rag)
                continue
            
            result = rag.query(question)
            
            print("\n" + "="*60)
            print("🤖 ANSWER:")
            print(result["answer"])
            
            if result.get("sources"):
                print(f"\n📚 Sources ({result['source_count']} documents):")
                for i, source in enumerate(result["sources"], 1):
                    content = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                    print(f"  [{i}] {content}")
        
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def show_info(rag):
    """Show system information."""
    info = rag.get_info()
    
    print("📊 System Information:")
    print(json.dumps(info, indent=2))


def main():
    """Main CLI function."""
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    # Initialize RAG system
    rag = SimpleRAGPipeline()
    
    if command == "help":
        print_help()
    
    elif command == "ingest":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide file paths to ingest")
            return
        
        file_paths = sys.argv[2:]
        ingest_files(rag, file_paths)
    
    elif command == "ingest-text":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide text to ingest")
            return
        
        text = " ".join(sys.argv[2:])
        ingest_text(rag, text)
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a question")
            return
        
        question = " ".join(sys.argv[2:])
        query(rag, question)
    
    elif command == "interactive":
        interactive_mode(rag)
    
    elif command == "info":
        show_info(rag)
    
    else:
        print(f"❌ Unknown command: {command}")
        print_help()


if __name__ == "__main__":
    main()