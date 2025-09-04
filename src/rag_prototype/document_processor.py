"""Document processing utilities for RAG prototype."""

import os
from typing import List, Union
from pathlib import Path
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

from .config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG system."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single document from file path."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine loader based on file extension
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
            else:
                # Try to load as text file for unknown extensions
                logger.warning(f"Unknown file extension {extension}, trying to load as text")
                loader = TextLoader(str(file_path), encoding='utf-8')
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def load_documents_from_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """Load all supported documents from a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        supported_extensions = {'.txt', '.pdf', '.docx', '.doc'}
        documents = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                docs = self.load_document(file_path)
                documents.extend(docs)
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            return []
    
    def process_text(self, text: str) -> List[Document]:
        """Process raw text into chunks."""
        document = Document(page_content=text, metadata={"source": "raw_text"})
        return self.chunk_documents([document])
    
    def process_documents(self, sources: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        Process documents from various sources.
        
        Args:
            sources: Can be a file path, directory path, or list of file paths
            
        Returns:
            List of chunked documents ready for embedding
        """
        if isinstance(sources, (str, Path)):
            sources = [sources]
        
        all_documents = []
        
        for source in sources:
            source_path = Path(source)
            
            if source_path.is_file():
                documents = self.load_document(source_path)
            elif source_path.is_dir():
                documents = self.load_documents_from_directory(source_path)
            else:
                logger.warning(f"Skipping invalid source: {source}")
                continue
                
            all_documents.extend(documents)
        
        # Chunk all documents
        return self.chunk_documents(all_documents)