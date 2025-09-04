"""Configuration management for RAG prototype."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for RAG prototype."""
    
    def __init__(self):
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        self.model_name: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.openai_api_key:
            print("Warning: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            return False
        return True


# Global config instance
config = Config()