"""Vector store implementation for RAG prototype."""

import os
from typing import List, Optional, Tuple
import logging
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from .config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for document embeddings using ChromaDB."""
    
    def __init__(self, use_openai_embeddings: bool = True):
        """
        Initialize vector store.
        
        Args:
            use_openai_embeddings: If True, use OpenAI embeddings. Otherwise use sentence-transformers.
        """
        self.db_path = config.chroma_db_path
        self.use_openai = use_openai_embeddings
        
        # Create data directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        if self.use_openai and config.openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=config.openai_api_key,
                model=config.embedding_model
            )
            logger.info(f"Using OpenAI embeddings with model: {config.embedding_model}")
        else:
            # Fallback to sentence-transformers
            self.embeddings = SentenceTransformersEmbeddings()
            logger.info("Using SentenceTransformer embeddings")
        
        # Initialize ChromaDB
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one."""
        try:
            # Try to load existing vectorstore
            if os.path.exists(self.db_path):
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vectorstore from {self.db_path}")
            else:
                # Create new vectorstore
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                logger.info(f"Created new vectorstore at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return []
            
            # Add documents and get IDs
            ids = self.vectorstore.add_documents(documents)
            
            # Persist the changes
            self.vectorstore.persist()
            
            logger.info(f"Added {len(documents)} documents to vectorstore")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {str(e)}")
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter_dict: Optional filter criteria
            
        Returns:
            List of similar documents
        """
        try:
            if not self.vectorstore:
                logger.error("Vectorstore not initialized")
                return []
            
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter_dict: Optional filter criteria
            
        Returns:
            List of tuples (document, similarity_score)
        """
        try:
            if not self.vectorstore:
                logger.error("Vectorstore not initialized")
                return []
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with score: {str(e)}")
            return []
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            if not self.vectorstore:
                return {"error": "Vectorstore not initialized"}
            
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "db_path": self.db_path,
                "embedding_model": "openai" if self.use_openai else "sentence-transformers"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")


class SentenceTransformersEmbeddings:
    """Custom embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SentenceTransformer model: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()