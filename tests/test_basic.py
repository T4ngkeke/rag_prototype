#!/usr/bin/env python3
"""
Simple test script for RAG prototype.
Tests basic functionality without requiring API keys.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_prototype.rag_pipeline import RAGPipeline
from rag_prototype.document_processor import DocumentProcessor
from rag_prototype.vector_store import VectorStore

def test_document_processor():
    """Test document processing functionality."""
    print("Testing Document Processor...")
    
    processor = DocumentProcessor()
    
    # Test text processing
    test_text = "This is a test document for the RAG system. It contains multiple sentences to test chunking."
    documents = processor.process_text(test_text)
    
    assert len(documents) > 0, "No documents created from text"
    assert documents[0].page_content, "Document content is empty"
    
    print("✅ Document processor test passed")

def test_vector_store():
    """Test vector store functionality."""
    print("Testing Vector Store...")
    
    # Use local embeddings for testing
    vector_store = VectorStore(use_openai_embeddings=False)
    
    # Test adding documents
    from langchain.schema import Document
    test_docs = [
        Document(page_content="Test content 1", metadata={"source": "test1"}),
        Document(page_content="Test content 2", metadata={"source": "test2"})
    ]
    
    ids = vector_store.add_documents(test_docs)
    assert len(ids) == 2, "Failed to add documents"
    
    # Test similarity search
    results = vector_store.similarity_search("test content", k=1)
    assert len(results) > 0, "No search results returned"
    
    print("✅ Vector store test passed")

def test_rag_pipeline():
    """Test complete RAG pipeline."""
    print("Testing RAG Pipeline...")
    
    # Use local embeddings for testing
    rag = RAGPipeline(use_openai_embeddings=False)
    
    # Test text ingestion
    test_text = """
    Python is a programming language. It was created by Guido van Rossum.
    Python is known for its simplicity and readability.
    It's widely used in data science and machine learning.
    """
    
    result = rag.ingest_text(test_text, {"source": "test"})
    assert result["success"], "Failed to ingest text"
    assert result["document_count"] > 0, "No documents created"
    
    # Test retrieval
    docs = rag.retrieve_relevant_docs("Python programming", k=2)
    assert len(docs) > 0, "No documents retrieved"
    
    # Test system info
    info = rag.get_system_info()
    assert "vectorstore" in info, "System info missing vectorstore"
    
    print("✅ RAG pipeline test passed")

def test_with_sample_files():
    """Test with actual sample files."""
    print("Testing with sample files...")
    
    rag = RAGPipeline(use_openai_embeddings=False)
    
    # Find sample files
    data_dir = Path(__file__).parent.parent / "data"
    sample_files = list(data_dir.glob("sample_*.txt"))
    
    if sample_files:
        # Ingest sample files
        result = rag.ingest_documents([str(f) for f in sample_files])
        assert result["success"], "Failed to ingest sample files"
        
        # Test query
        query_result = rag.query("What is Python?", return_sources=True)
        assert query_result["answer"], "No answer generated"
        assert query_result["source_count"] > 0, "No sources found"
        
        print("✅ Sample files test passed")
    else:
        print("⚠️  No sample files found, skipping test")

def main():
    """Run all tests."""
    print("🧪 Running RAG Prototype Tests")
    print("=" * 40)
    
    try:
        test_document_processor()
        test_vector_store()
        test_rag_pipeline()
        test_with_sample_files()
        
        print("\n🎉 All tests passed!")
        print("\nThe RAG prototype is working correctly.")
        print("You can now use the system with confidence.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()