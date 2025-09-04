"""Streamlit web interface for RAG prototype."""

import streamlit as st
import json
from pathlib import Path
import tempfile
import os

# Add src to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag_prototype.rag_pipeline import RAGPipeline
from rag_prototype.config import config

# Page configuration
st.set_page_config(
    page_title="RAG Prototype",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag():
    """Initialize RAG pipeline."""
    use_openai = st.sidebar.checkbox("Use OpenAI Embeddings", value=True)
    
    if st.session_state.rag_pipeline is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_pipeline = RAGPipeline(use_openai_embeddings=use_openai)
    
    return st.session_state.rag_pipeline

def main():
    """Main Streamlit application."""
    st.title("🤖 RAG Prototype")
    st.markdown("*Retrieval-Augmented Generation System*")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Check API key
    api_key_status = "✅ Set" if config.openai_api_key else "❌ Not set"
    st.sidebar.info(f"OpenAI API Key: {api_key_status}")
    
    # Initialize RAG
    rag = initialize_rag()
    
    # System info in sidebar
    if st.sidebar.button("Show System Info"):
        info = rag.get_system_info()
        st.sidebar.json(info)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Document Management", "ℹ️ System Info"])
    
    with tab1:
        st.header("Chat with your documents")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Assistant:** {answer}")
                
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} documents)"):
                        for j, source in enumerate(sources, 1):
                            st.markdown(f"**[{j}]** {source['content'][:300]}...")
                            if source.get('metadata'):
                                st.caption(f"Source: {source['metadata']}")
                
                st.divider()
        
        # Question input
        question = st.text_input("Ask a question:", key="question_input")
        col1, col2 = st.columns([1, 4])
        
        with col1:
            k = st.number_input("Number of sources", min_value=1, max_value=10, value=4)
        
        if st.button("Ask Question", type="primary") and question:
            with st.spinner("Searching and generating answer..."):
                result = rag.query(question, k=k, return_sources=True)
                
                # Add to chat history
                sources = result.get('sources', [])
                st.session_state.chat_history.append((question, result['answer'], sources))
                
                # Rerun to update display
                st.rerun()
    
    with tab2:
        st.header("Document Management")
        
        # File upload
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing uploaded files..."):
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                # Save uploaded files temporarily
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Ingest documents
                result = rag.ingest_documents(file_paths)
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.info(f"📊 Processed {result['document_count']} document chunks")
                else:
                    st.error(f"❌ {result['message']}")
                
                # Clean up temp files
                import shutil
                shutil.rmtree(temp_dir)
        
        st.divider()
        
        # Text input
        st.subheader("📝 Add Text Directly")
        text_input = st.text_area("Enter text to add to the knowledge base:", height=150)
        source_name = st.text_input("Source name (optional):", value="Direct Input")
        
        if st.button("Add Text") and text_input:
            with st.spinner("Processing text..."):
                metadata = {"source": source_name} if source_name else {}
                result = rag.ingest_text(text_input, metadata=metadata)
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.info(f"📊 Created {result['document_count']} text chunks")
                else:
                    st.error(f"❌ {result['message']}")
    
    with tab3:
        st.header("System Information")
        
        if st.button("Refresh Info"):
            info = rag.get_system_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Vector Store")
                vectorstore_info = info.get('vectorstore', {})
                st.metric("Document Count", vectorstore_info.get('document_count', 0))
                st.text(f"DB Path: {vectorstore_info.get('db_path', 'N/A')}")
                st.text(f"Embeddings: {vectorstore_info.get('embedding_model', 'N/A')}")
            
            with col2:
                st.subheader("🔧 Configuration")
                st.text(f"LLM Model: {info.get('llm_model', 'N/A')}")
                st.text(f"Embedding Model: {info.get('embedding_model', 'N/A')}")
                st.text(f"Chunk Size: {info.get('chunk_size', 'N/A')}")
                st.text(f"Chunk Overlap: {info.get('chunk_overlap', 'N/A')}")
            
            st.subheader("🔧 Full Configuration")
            st.json(info)

if __name__ == "__main__":
    main()