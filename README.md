# 🤖 RAG Prototype

A simple yet powerful Retrieval-Augmented Generation (RAG) system built with Python. This prototype demonstrates how to build a complete RAG pipeline that can ingest documents, create embeddings, perform similarity search, and generate contextual answers using Large Language Models.

## 🚀 Features

- **Document Processing**: Support for PDF, DOCX, and TXT files
- **Vector Search**: ChromaDB-based vector storage with similarity search
- **Flexible Embeddings**: Choose between OpenAI embeddings or local sentence-transformers
- **LLM Integration**: OpenAI GPT models for answer generation
- **Multiple Interfaces**: CLI, Web UI (Streamlit), and Python API
- **Easy Configuration**: Environment-based configuration management

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional, for OpenAI embeddings and LLM)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag_prototype
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env file with your OpenAI API key
   ```

## ⚙️ Configuration

The system can be configured through environment variables. Copy `.env.example` to `.env` and modify as needed:

```bash
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_DB_PATH=./data/chroma_db
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4000
TEMPERATURE=0.1
```

**Note**: The system can work without an OpenAI API key by using local embeddings and disabling LLM generation.

## 🚀 Quick Start

### 1. Run the Example Script

```bash
python examples/example_usage.py
```

This will demonstrate the basic functionality using sample documents.

### 2. Command Line Interface

**Ingest documents**:
```bash
python -m src.rag_prototype.cli ingest data/sample_ai_document.txt data/sample_python_document.txt
```

**Ask questions**:
```bash
python -m src.rag_prototype.cli query "What is artificial intelligence?"
```

**Interactive mode**:
```bash
python -m src.rag_prototype.cli interactive
```

**Show system info**:
```bash
python -m src.rag_prototype.cli info
```

### 3. Web Interface

Launch the Streamlit web application:

```bash
streamlit run src/rag_prototype/streamlit_app.py
```

Then open your browser to `http://localhost:8501` to access the web interface.

## 📖 Usage Examples

### Python API

```python
from src.rag_prototype.rag_pipeline import RAGPipeline

# Initialize RAG system
rag = RAGPipeline(use_openai_embeddings=True)

# Ingest documents
result = rag.ingest_documents(["path/to/document.pdf"])
print(f"Ingested {result['document_count']} chunks")

# Ask questions
answer = rag.query("What is the main topic of the document?")
print(answer["answer"])
```

### CLI Commands

```bash
# Ingest a directory of documents
python -m src.rag_prototype.cli ingest /path/to/documents/

# Ingest raw text
python -m src.rag_prototype.cli ingest-text "Your text content here"

# Query with custom parameters
python -m src.rag_prototype.cli query "Your question" --k 5 --json-output

# Check configuration
python -m src.rag_prototype.cli config-info
```

## 🏗️ Architecture

The RAG prototype consists of several key components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Document       │    │  Vector Store    │    │  LLM            │
│  Processor      │───▶│  (ChromaDB)      │───▶│  (OpenAI GPT)   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Text Chunking  │    │  Similarity      │    │  Answer         │
│  & Embedding    │    │  Search          │    │  Generation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components

1. **Document Processor** (`document_processor.py`):
   - Loads and processes various document formats
   - Splits documents into manageable chunks
   - Handles metadata extraction

2. **Vector Store** (`vector_store.py`):
   - Manages document embeddings using ChromaDB
   - Supports both OpenAI and local embeddings
   - Provides similarity search functionality

3. **RAG Pipeline** (`rag_pipeline.py`):
   - Orchestrates the complete RAG workflow
   - Combines retrieval and generation
   - Manages the query processing pipeline

4. **Interfaces**:
   - CLI (`cli.py`): Command-line interface
   - Web UI (`streamlit_app.py`): Streamlit web application
   - Python API: Direct programmatic access

## 🔧 Customization

### Using Local Embeddings

If you don't have an OpenAI API key, you can use local embeddings:

```bash
python -m src.rag_prototype.cli ingest --local-embeddings document.txt
python -m src.rag_prototype.cli query --local-embeddings "Your question"
```

### Custom Configuration

Modify the configuration in `src/rag_prototype/config.py` or use environment variables to customize:

- Chunk size and overlap
- Model selection
- Temperature and token limits
- Database paths

## 📁 Project Structure

```
rag_prototype/
├── src/rag_prototype/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── document_processor.py  # Document loading and processing
│   ├── vector_store.py     # Vector storage and search
│   ├── rag_pipeline.py     # Main RAG pipeline
│   ├── cli.py             # Command-line interface
│   ├── streamlit_app.py   # Web interface
│   └── main.py            # Entry point
├── data/
│   ├── sample_ai_document.txt
│   ├── sample_python_document.txt
│   └── chroma_db/         # Vector database (created automatically)
├── examples/
│   └── example_usage.py   # Usage examples
├── requirements.txt       # Python dependencies
├── .env.example          # Configuration template
├── .gitignore
└── README.md
```

## 🧪 Testing

Run the example script to test the system:

```bash
python examples/example_usage.py
```

This will:
1. Initialize the RAG system
2. Ingest sample documents
3. Run example queries
4. Display system information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running commands from the project root directory
2. **API key issues**: Set your OpenAI API key in the `.env` file or use local embeddings
3. **Memory issues**: Reduce chunk size if processing large documents
4. **Port conflicts**: Use a different port for Streamlit if 8501 is occupied

### Getting Help

- Check the example usage script for working code
- Review the CLI help: `python -m src.rag_prototype.cli --help`
- Examine the logs for detailed error information

## 🔮 Future Enhancements

- [ ] Support for more document formats (HTML, Markdown, etc.)
- [ ] Integration with more LLM providers (Anthropic, Hugging Face)
- [ ] Advanced retrieval strategies (hybrid search, re-ranking)
- [ ] Chat history and conversation memory
- [ ] Batch processing capabilities
- [ ] REST API interface
- [ ] Docker containerization
- [ ] Performance monitoring and analytics
