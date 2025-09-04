#!/usr/bin/env python3
"""
Setup script for RAG prototype.
Provides an easy way to test the system and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_banner():
    """Print setup banner."""
    print("🤖 RAG Prototype Setup")
    print("=" * 40)


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def test_simple_version():
    """Test the simple version that doesn't require dependencies."""
    print("\n🧪 Testing simple RAG implementation...")
    
    try:
        # Import and run simple test
        from simple_rag import SimpleRAGPipeline
        
        rag = SimpleRAGPipeline()
        
        # Test text ingestion
        result = rag.ingest_text("Test content for RAG system", {"source": "setup_test"})
        if result["success"]:
            print("✅ Text ingestion works")
        else:
            print("❌ Text ingestion failed")
            return False
        
        # Test query
        query_result = rag.query("test content")
        if query_result["source_count"] > 0:
            print("✅ Query and retrieval works")
        else:
            print("❌ Query failed")
            return False
        
        print("✅ Simple RAG implementation is working!")
        return True
        
    except Exception as e:
        print(f"❌ Simple RAG test failed: {str(e)}")
        return False


def install_dependencies():
    """Try to install dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Try to install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError:
        print("⚠️  Could not install all dependencies")
        print("   You can still use the simple version without external dependencies")
        return False
    
    except Exception as e:
        print(f"⚠️  Installation error: {str(e)}")
        return False


def test_full_version():
    """Test the full version with dependencies."""
    print("\n🧪 Testing full RAG implementation...")
    
    try:
        # Try to import langchain components
        from src.rag_prototype.rag_pipeline import RAGPipeline
        
        print("✅ Full RAG imports successful")
        
        # Test initialization (without OpenAI key)
        rag = RAGPipeline(use_openai_embeddings=False)
        print("✅ Full RAG initialization successful")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Import error: {str(e)}")
        print("   Some dependencies are missing")
        return False
    
    except Exception as e:
        print(f"⚠️  Full RAG test failed: {str(e)}")
        return False


def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print("-" * 30)
    
    print("\n🔧 Simple Version (no dependencies):")
    print("  python simple_rag.py                    # Run demo")
    print("  python simple_cli.py help               # Show CLI help")
    print("  python simple_cli.py query 'question'   # Ask a question")
    print("  python simple_cli.py interactive        # Interactive mode")
    
    print("\n🚀 Full Version (with dependencies):")
    print("  python examples/example_usage.py        # Run example")
    print("  python -m src.rag_prototype.cli help    # Show CLI help")
    print("  streamlit run src/rag_prototype/streamlit_app.py  # Web interface")
    
    print("\n⚙️  Configuration:")
    print("  cp .env.example .env                     # Copy config template")
    print("  # Edit .env file with your OpenAI API key")


def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Test simple version first
    simple_works = test_simple_version()
    
    # Try to install dependencies
    deps_installed = install_dependencies()
    
    # Test full version if dependencies are available
    full_works = False
    if deps_installed:
        full_works = test_full_version()
    
    # Summary
    print("\n📋 Setup Summary:")
    print("-" * 20)
    print(f"✅ Simple RAG: {'Working' if simple_works else 'Failed'}")
    print(f"📦 Dependencies: {'Installed' if deps_installed else 'Failed/Skipped'}")
    print(f"🚀 Full RAG: {'Working' if full_works else 'Not available'}")
    
    if simple_works:
        print("\n🎉 RAG Prototype is ready to use!")
        
        if not deps_installed:
            print("\n💡 For full functionality, install dependencies manually:")
            print("   pip install -r requirements.txt")
        
        show_usage_examples()
    else:
        print("\n❌ Setup failed. Please check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()