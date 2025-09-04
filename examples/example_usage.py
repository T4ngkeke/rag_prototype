#!/usr/bin/env python3
"""
Example usage script for RAG prototype.
This script demonstrates how to use the RAG system programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_prototype.rag_pipeline import RAGPipeline
from rag_prototype.config import config

def main():
    """Demonstrate RAG prototype usage."""
    print("🤖 RAG Prototype Example")
    print("=" * 50)
    
    # Check configuration
    print(f"OpenAI API Key: {'✅ Set' if config.openai_api_key else '❌ Not set'}")
    
    # Initialize RAG pipeline
    print("\n🔧 Initializing RAG pipeline...")
    # Use local embeddings if no OpenAI key
    use_openai = bool(config.openai_api_key)
    rag = RAGPipeline(use_openai_embeddings=use_openai)
    
    # Ingest sample documents
    print("\n📚 Ingesting sample documents...")
    data_dir = Path(__file__).parent.parent / "data"
    sample_docs = [
        data_dir / "sample_ai_document.txt",
        data_dir / "sample_python_document.txt"
    ]
    
    # Filter existing files
    existing_docs = [str(doc) for doc in sample_docs if doc.exists()]
    
    if existing_docs:
        result = rag.ingest_documents(existing_docs)
        if result["success"]:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ {result['message']}")
    else:
        print("❌ No sample documents found")
        return
    
    # Example queries
    example_questions = [
        "What is artificial intelligence?",
        "What are the key features of Python?",
        "How does machine learning work?",
        "What are Python data types?",
        "What are some popular Python libraries?"
    ]
    
    print("\n❓ Example queries:")
    print("-" * 30)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n[{i}] Question: {question}")
        
        # Process query
        result = rag.query(question, k=2, return_sources=False)
        answer = result["answer"]
        
        # Display answer (truncated)
        if len(answer) > 200:
            answer = answer[:200] + "..."
        
        print(f"    Answer: {answer}")
        print(f"    Sources: {result['source_count']} documents")
    
    # System information
    print(f"\n📊 System Information:")
    info = rag.get_system_info()
    print(f"    Documents in vector store: {info['vectorstore'].get('document_count', 0)}")
    print(f"    LLM Model: {info.get('llm_model', 'None')}")
    print(f"    Embedding Model: {info.get('embedding_model', 'None')}")
    
    print("\n✅ Example completed!")
    print("\nTo interact with the system:")
    print("  CLI: python -m rag_prototype.cli interactive")
    print("  Web: streamlit run src/rag_prototype/streamlit_app.py")

if __name__ == "__main__":
    main()