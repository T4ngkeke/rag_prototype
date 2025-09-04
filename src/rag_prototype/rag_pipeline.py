"""RAG (Retrieval-Augmented Generation) pipeline implementation."""

from typing import List, Dict, Any, Optional, Tuple
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

from .config import config
from .vector_store import VectorStore
from .document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for question answering."""
    
    def __init__(self, use_openai_embeddings: bool = True):
        """
        Initialize RAG pipeline.
        
        Args:
            use_openai_embeddings: Whether to use OpenAI embeddings
        """
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(use_openai_embeddings=use_openai_embeddings)
        
        # Initialize LLM
        if config.openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=config.openai_api_key,
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            logger.info(f"Initialized OpenAI LLM: {config.model_name}")
        else:
            self.llm = None
            logger.warning("No OpenAI API key provided. LLM functionality disabled.")
        
        # Initialize prompt templates
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for RAG."""
        # RAG prompt template
        self.rag_template = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based only on the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but thorough in your response
4. If you're uncertain about any part of the answer, indicate that uncertainty

Answer:"""

        self.rag_prompt = PromptTemplate(
            template=self.rag_template,
            input_variables=["context", "question"]
        )
        
        # Chat prompt template
        self.chat_template = ChatPromptTemplate.from_template(self.rag_template)
    
    def ingest_documents(self, sources: List[str]) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            sources: List of file paths or directory paths
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Starting document ingestion from {len(sources)} sources")
            
            # Process documents
            documents = self.document_processor.process_documents(sources)
            
            if not documents:
                return {
                    "success": False,
                    "message": "No documents were processed",
                    "document_count": 0
                }
            
            # Add to vector store
            ids = self.vector_store.add_documents(documents)
            
            return {
                "success": True,
                "message": f"Successfully ingested {len(documents)} document chunks",
                "document_count": len(documents),
                "chunk_ids": ids
            }
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {str(e)}")
            return {
                "success": False,
                "message": f"Error during ingestion: {str(e)}",
                "document_count": 0
            }
    
    def ingest_text(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Ingest raw text into the RAG system.
        
        Args:
            text: Raw text to ingest
            metadata: Optional metadata for the text
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Process text into chunks
            documents = self.document_processor.process_text(text)
            
            # Add metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # Add to vector store
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
                "message": f"Error during text ingestion: {str(e)}",
                "document_count": 0
            }
    
    def retrieve_relevant_docs(
        self, 
        query: str, 
        k: int = 4,
        include_scores: bool = False
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            include_scores: Whether to include similarity scores
            
        Returns:
            List of relevant documents
        """
        try:
            if include_scores:
                results = self.vector_store.similarity_search_with_score(query, k=k)
                return [(doc, score) for doc, score in results]
            else:
                return self.vector_store.similarity_search(query, k=k)
                
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate an answer using the LLM and retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            
        Returns:
            Generated answer
        """
        if not self.llm:
            return "Error: No LLM available. Please provide an OpenAI API key."
        
        try:
            # Combine context documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(context_docs)
            ])
            
            # Create prompt
            prompt = self.rag_prompt.format(context=context, question=query)
            
            # Generate response
            response = self.llm.predict(prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def query(
        self, 
        question: str, 
        k: int = 4,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG query pipeline.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing RAG query: {question[:100]}...")
            
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_docs(question, k=k)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "source_count": 0
                }
            
            # Generate answer
            answer = self.generate_answer(question, relevant_docs)
            
            # Prepare response
            response = {
                "answer": answer,
                "source_count": len(relevant_docs)
            }
            
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ]
            
            return response
            
        except Exception as e:
            logger.error(f"Error during RAG query: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "source_count": 0
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        vector_info = self.vector_store.get_collection_info()
        
        return {
            "vectorstore": vector_info,
            "llm_model": config.model_name if self.llm else "None",
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }