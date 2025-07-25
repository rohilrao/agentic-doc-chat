"""
Streamlit Frontend for Agentic RAG System
Supports document upload, URL ingestion, and chat interface.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import List

# Import the modular RAG system
from agentic_rag import AgenticRAG, DocumentProcessor

# Set page config
st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

def initialize_rag_system():
    """Initialize the RAG system with user settings."""
    try:
        # Store settings in session state for access
        st.session_state.rag_system = AgenticRAG(
            chat_model=st.session_state.get('chat_model', 'llama3.1:8b'),
            embedding_model=st.session_state.get('embedding_model', 'nomic-embed-text'),
            ollama_base_url=st.session_state.get('ollama_url', 'http://localhost:11434'),
            chunk_size=st.session_state.get('chunk_size', 1000),
            chunk_overlap=st.session_state.get('chunk_overlap', 200)
        )
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return False

def load_documents():
    """Load documents into the RAG system."""
    if st.session_state.rag_system is None:
        st.error("Please initialize the RAG system first")
        return
    
    # This would be replaced with actual document loading
    # st.session_state.rag_system.add_documents(...)
    st.session_state.documents_loaded = True

def main():
    st.title("🤖 Agentic RAG Chat System")
    st.markdown("Upload documents, add URLs, and chat with your AI assistant!")
    
    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        chat_model = st.selectbox(
            "Chat Model",
            ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "codellama:7b"],
            index=0
        )
        
        embedding_model = st.selectbox(
            "Embedding Model", 
            ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
            index=0
        )
        
        ollama_url = st.text_input(
            "Ollama Server URL", 
            value="http://localhost:11434"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
        
        # Initialize system
        if st.button("Initialize RAG System", type="primary"):
            # Store current settings in session state
            st.session_state.chat_model = chat_model
            st.session_state.embedding_model = embedding_model
            st.session_state.ollama_url = ollama_url
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            
            with st.spinner("Initializing RAG system..."):
                if initialize_rag_system():
                    st.success("RAG system initialized!")
                else:
                    st.error("Failed to initialize RAG system")
        
        st.divider()
        
        # Document Management
        st.header("📚 Document Management")
        
        # File upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'doc', 'csv'],
            help="Supported formats: PDF, TXT, DOCX, DOC, CSV"
        )
        
        # URL input
        st.subheader("Add URLs")
        urls_text = st.text_area(
            "Enter URLs (one per line)",
            placeholder="https://example.com/article1\nhttps://example.com/article2"
        )
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        # Load documents button
        if st.button("Load Documents", type="secondary"):
            if not st.session_state.rag_system or st.session_state.rag_system == "placeholder":
                st.error("Please initialize the RAG system first!")
                return
                
            if uploaded_files or urls:
                with st.spinner("Loading documents..."):
                    try:
                        # Actually load documents using the RAG system
                        chunks_added = st.session_state.rag_system.add_documents(
                            uploaded_files=uploaded_files,
                            urls=urls
                        )
                        
                        # Build the graph after adding documents
                        st.session_state.rag_system.build_graph()
                        
                        # Update session state
                        st.session_state.documents_loaded = True
                        st.session_state.indexed_docs = st.session_state.rag_system.get_indexed_documents()
                        
                        st.success(f"Successfully loaded {chunks_added} document chunks!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading documents: {e}")
            else:
                st.warning("Please upload files or add URLs first")
        
        # Show indexed documents
        if st.session_state.indexed_docs:
            st.subheader("Indexed Documents")
            for doc in st.session_state.indexed_docs:
                st.text(f"• {doc}")
            
            if st.button("Clear Knowledge Base"):
                if st.session_state.rag_system and st.session_state.rag_system != "placeholder":
                    st.session_state.rag_system.clear_knowledge_base()
                st.session_state.indexed_docs = []
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.success("Knowledge base cleared!")
                st.rerun()
    
    # Main chat interface
    if not st.session_state.documents_loaded:
        st.info("👆 Please configure the system and load some documents to start chatting!")
        
        # Show example of supported file types
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📄 Supported Files")
            st.markdown("""
            - PDF documents
            - Text files (.txt)
            - Word documents (.docx, .doc)
            - CSV files
            """)
        
        with col2:
            st.subheader("🌐 Web Content")
            st.markdown("""
            - Blog posts
            - Articles
            - Documentation
            - Any web page content
            """)
        
        with col3:
            st.subheader("🤖 AI Features")
            st.markdown("""
            - Relevance checking
            - Query rewriting
            - Context-aware responses
            - Chat memory
            """)
    
    else:
        # Chat interface
        st.header("💬 Chat with your Documents")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(message)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.chat_history.append(("user", prompt))
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Use the actual RAG system to generate response
                        if st.session_state.rag_system and st.session_state.rag_system != "placeholder":
                            response = st.session_state.rag_system.query(prompt, verbose=False)
                        else:
                            response = "❌ RAG system not initialized. Please initialize it in the sidebar first."
                    except Exception as e:
                        response = f"❌ Error generating response: {str(e)}"
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append(("assistant", response))
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Show system status
        with st.expander("System Status"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents Loaded", len(st.session_state.indexed_docs))
                st.metric("Chat Messages", len(st.session_state.chat_history))
            with col2:
                st.write("**Model Configuration:**")
                st.text(f"Chat: {chat_model}")
                st.text(f"Embeddings: {embedding_model}")
                st.text(f"Server: {ollama_url}")

# Instructions for running
def show_instructions():
    """Show setup instructions."""
    with st.expander("🚀 Setup Instructions", expanded=False):
        st.markdown("""
        ### Prerequisites
        
        1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
        
        2. **Pull required models**:
           ```bash
           ollama pull llama3.1:8b
           ollama pull nomic-embed-text
           ```
        
        3. **Install Python dependencies**:
           ```bash
           pip install streamlit langchain langgraph langchain-community 
           pip install langchain-ollama chromadb beautifulsoup4 
           pip install langchain-text-splitters pypdf python-docx unstructured
           ```
        
        4. **Start Ollama server** (if not running as service):
           ```bash
           ollama serve
           ```
        
        5. **Run this Streamlit app**:
           ```bash
           streamlit run streamlit_app.py
           ```
        
        ### Usage
        1. Configure your models in the sidebar
        2. Initialize the RAG system
        3. Upload documents or add URLs
        4. Start chatting with your AI assistant!
        """)

if __name__ == "__main__":
    show_instructions()
    main()