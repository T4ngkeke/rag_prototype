"""
Simplified RAG prototype that works without external dependencies.
This version demonstrates the core concepts using basic implementations.
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleConfig:
    """Simple configuration without external dependencies."""
    
    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.db_path = os.getenv("DB_PATH", "./data/simple_db")
        
        # Create data directory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


class SimpleDocument:
    """Simple document representation."""
    
    def __init__(self, content: str, metadata: Dict = None):
        self.page_content = content
        self.metadata = metadata or {}


class SimpleTextSplitter:
    """Simple text splitter without external dependencies."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def split_documents(self, documents: List[SimpleDocument]) -> List[SimpleDocument]:
        """Split documents into chunks."""
        chunks = []
        
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunks.append(SimpleDocument(chunk, chunk_metadata))
        
        return chunks


class SimpleDocumentProcessor:
    """Simple document processor."""
    
    def __init__(self):
        self.text_splitter = SimpleTextSplitter()
    
    def load_text_file(self, file_path: str) -> SimpleDocument:
        """Load a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                'source': file_path,
                'filename': os.path.basename(file_path)
            }
            
            return SimpleDocument(content, metadata)
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def process_documents(self, file_paths: List[str]) -> List[SimpleDocument]:
        """Process multiple documents."""
        documents = []
        
        for file_path in file_paths:
            if Path(file_path).suffix.lower() == '.txt':
                doc = self.load_text_file(file_path)
                if doc:
                    documents.append(doc)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
        
        return self.text_splitter.split_documents(documents)
    
    def process_text(self, text: str, metadata: Dict = None) -> List[SimpleDocument]:
        """Process raw text."""
        doc = SimpleDocument(text, metadata or {})
        return self.text_splitter.split_documents([doc])


class SimpleVectorStore:
    """Simple vector store using basic similarity."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.documents = {}
        self.index_file = os.path.join(db_path, "index.pkl")
        
        # Create directory
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing index."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from index")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                self.documents = {}
    
    def _save_index(self):
        """Save index to disk."""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def add_documents(self, documents: List[SimpleDocument]) -> List[str]:
        """Add documents to the store."""
        ids = []
        
        for doc in documents:
            # Create a simple ID based on content hash
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            self.documents[doc_id] = {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            
            ids.append(doc_id)
        
        self._save_index()
        logger.info(f"Added {len(documents)} documents to store")
        return ids
    
    def simple_similarity_search(self, query: str, k: int = 4) -> List[SimpleDocument]:
        """Simple keyword-based similarity search."""
        query_words = set(query.lower().split())
        scores = []
        
        for doc_id, doc_data in self.documents.items():
            content = doc_data['content'].lower()
            content_words = set(content.split())
            
            # Simple overlap score
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                scores.append((doc_id, overlap, doc_data))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score, doc_data in scores[:k]:
            doc = SimpleDocument(doc_data['content'], doc_data['metadata'])
            results.append(doc)
        
        return results
    
    def get_count(self) -> int:
        """Get document count."""
        return len(self.documents)


class SimpleRAGPipeline:
    """Simple RAG pipeline without LLM integration."""
    
    def __init__(self, db_path: str = "./data/simple_db"):
        self.config = SimpleConfig()
        self.document_processor = SimpleDocumentProcessor()
        self.vector_store = SimpleVectorStore(db_path)
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest documents into the system."""
        try:
            documents = self.document_processor.process_documents(file_paths)
            
            if not documents:
                return {
                    "success": False,
                    "message": "No documents processed",
                    "document_count": 0
                }
            
            ids = self.vector_store.add_documents(documents)
            
            return {
                "success": True,
                "message": f"Successfully ingested {len(documents)} document chunks",
                "document_count": len(documents),
                "chunk_ids": ids
            }
        
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "document_count": 0
            }
    
    def ingest_text(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        """Ingest raw text."""
        try:
            documents = self.document_processor.process_text(text, metadata)
            ids = self.vector_store.add_documents(documents)
            
            return {
                "success": True,
                "message": f"Successfully ingested text as {len(documents)} chunks",
                "document_count": len(documents),
                "chunk_ids": ids
            }
        
        except Exception as e:
            logger.error(f"Error during text ingestion: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "document_count": 0
            }
    
    def search(self, query: str, k: int = 4) -> List[SimpleDocument]:
        """Search for relevant documents."""
        return self.vector_store.simple_similarity_search(query, k=k)
    
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Process a query and return results."""
        try:
            # Search for relevant documents
            docs = self.search(question, k=k)
            
            if not docs:
                return {
                    "answer": "No relevant documents found for your query.",
                    "sources": [],
                    "source_count": 0
                }
            
            # Simple answer generation (concatenate relevant content)
            context_parts = []
            sources = []
            
            for i, doc in enumerate(docs):
                context_parts.append(f"[{i+1}] {doc.page_content}")
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata
                })
            
            answer = f"Based on the available documents, here are the most relevant excerpts about '{question}':\n\n" + "\n\n".join(context_parts)
            
            return {
                "answer": answer,
                "sources": sources,
                "source_count": len(docs)
            }
        
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "source_count": 0
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "document_count": self.vector_store.get_count(),
            "db_path": self.vector_store.db_path,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap
        }


def main():
    """Simple demo of the RAG system."""
    print("🤖 Simple RAG Prototype")
    print("=" * 40)
    
    # Initialize system
    rag = SimpleRAGPipeline()
    
    # Add sample text
    sample_text = """
    Python is a high-level programming language created by Guido van Rossum.
    It was first released in 1991 and has become one of the most popular programming languages.
    Python is known for its simple syntax and readability.
    It's widely used in web development, data science, artificial intelligence, and automation.
    """
    
    print("📝 Adding sample text...")
    result = rag.ingest_text(sample_text, {"source": "sample"})
    print(f"✅ {result['message']}")
    
    # Try to ingest sample files if they exist
    data_dir = Path("./data")
    if data_dir.exists():
        sample_files = list(data_dir.glob("*.txt"))
        if sample_files:
            print(f"📚 Ingesting {len(sample_files)} sample files...")
            file_result = rag.ingest_documents([str(f) for f in sample_files])
            print(f"✅ {file_result['message']}")
    
    # Example queries
    queries = [
        "What is Python?",
        "Who created Python?",
        "What is Python used for?"
    ]
    
    print("\n❓ Example queries:")
    print("-" * 30)
    
    for query in queries:
        print(f"\nQ: {query}")
        result = rag.query(query, k=2)
        
        # Show truncated answer
        answer = result['answer']
        if len(answer) > 200:
            answer = answer[:200] + "..."
        
        print(f"A: {answer}")
        print(f"Sources: {result['source_count']} documents")
    
    # System info
    info = rag.get_info()
    print(f"\n📊 System info:")
    print(f"  Documents: {info['document_count']}")
    print(f"  Database: {info['db_path']}")
    
    print("\n✅ Simple RAG prototype is working!")


if __name__ == "__main__":
    main()