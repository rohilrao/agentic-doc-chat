"""
Modular LangGraph-based Agentic RAG System with Streamlit Support
Supports multiple document types and provides a clean interface for frontends.
"""

import os
import tempfile
from typing import Annotated, Literal, Sequence, List, Optional, Union
from typing_extensions import TypedDict
from pathlib import Path

# Core LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_core.documents import Document

# Ollama and community imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph imports
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Pydantic for structured output
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """State object that gets passed between nodes in the graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class DocumentRelevanceGrade(BaseModel):
    """Binary score for document relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class DocumentProcessor:
    """Handles loading and processing of different document types."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF',
        '.txt': 'Text',
        '.docx': 'Word Document',
        '.doc': 'Word Document',
        '.csv': 'CSV'
    }
    
    @classmethod
    def get_supported_types(cls) -> dict:
        """Get supported file types."""
        return cls.SUPPORTED_EXTENSIONS.copy()
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file type is supported."""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """Load a document based on its file type."""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            return loader.load()
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")
    
    @classmethod
    def load_from_urls(cls, urls: List[str]) -> List[Document]:
        """Load documents from URLs."""
        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Failed to load {url}: {str(e)}")
        return docs


class AgenticRAG:
    """Modular Agentic RAG implementation using Ollama models."""
    
    def __init__(self, 
                 chat_model: str = "llama3.1:8b",
                 embedding_model: str = "nomic-embed-text",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the Agentic RAG system.
        
        Args:
            chat_model: Ollama chat model name
            embedding_model: Ollama embedding model name  
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            ollama_base_url: Base URL for Ollama server
        """
        self.chat_model_name = chat_model
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_base_url = ollama_base_url
        
        # Initialize models with base URL
        self.chat_model = ChatOllama(
            model=chat_model, 
            temperature=0,
            base_url=ollama_base_url
        )
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )
        
        # Initialize components
        self.retriever = None
        self.retriever_tool = None
        self.graph = None
        self.vectorstore = None
        self.indexed_documents = []
        
    def add_documents(self, 
                     file_paths: Optional[List[str]] = None,
                     urls: Optional[List[str]] = None,
                     uploaded_files: Optional[List] = None) -> int:
        """
        Add documents to the knowledge base from various sources.
        
        Args:
            file_paths: List of local file paths
            urls: List of URLs to scrape
            uploaded_files: List of uploaded file objects (for Streamlit)
            
        Returns:
            int: Number of document chunks added
        """
        all_docs = []
        
        # Load from file paths
        if file_paths:
            for file_path in file_paths:
                if DocumentProcessor.is_supported(file_path):
                    try:
                        docs = DocumentProcessor.load_document(file_path)
                        all_docs.extend(docs)
                        self.indexed_documents.append(f"File: {Path(file_path).name}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        # Load from URLs
        if urls:
            try:
                docs = DocumentProcessor.load_from_urls(urls)
                all_docs.extend(docs)
                self.indexed_documents.extend([f"URL: {url}" for url in urls])
            except Exception as e:
                print(f"Error loading URLs: {e}")
        
        # Handle uploaded files (for Streamlit)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Load the temporary file
                    docs = DocumentProcessor.load_document(tmp_path)
                    all_docs.extend(docs)
                    self.indexed_documents.append(f"Upload: {uploaded_file.name}")
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    print(f"Error processing uploaded file {uploaded_file.name}: {e}")
        
        if not all_docs:
            raise ValueError("No documents were successfully loaded")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(all_docs)
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding=self.embeddings,
            )
        else:
            # Add to existing vectorstore
            self.vectorstore.add_documents(doc_splits)
        
        self.retriever = self.vectorstore.as_retriever()
        
        # Create retriever tool
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_documents",
            "Search and return information from the indexed documents.",
        )
        
        print(f"Added {len(doc_splits)} document chunks to knowledge base")
        return len(doc_splits)
    
    def get_indexed_documents(self) -> List[str]:
        """Get list of indexed documents."""
        return self.indexed_documents.copy()
    
    def clear_knowledge_base(self):
        """Clear all indexed documents."""
        self.vectorstore = None
        self.retriever = None
        self.retriever_tool = None
        self.indexed_documents = []
        self.graph = None
        print("Knowledge base cleared")
    
    def grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """Check if retrieved documents are relevant to the question."""
        print("---CHECK RELEVANCE---")
        
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        grading_model = ChatOllama(
            model=self.chat_model_name,
            temperature=0,
            base_url=self.ollama_base_url
        )
        llm_with_tool = grading_model.with_structured_output(DocumentRelevanceGrade)
        
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of retrieved documents to a user question.

Retrieved documents:
{context}

User question: {question}

If the documents contain keywords or semantic meaning related to the question, grade as relevant.
Give a binary score 'yes' or 'no'.""",
            input_variables=["context", "question"],
        )
        
        chain = prompt | llm_with_tool
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
        
        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"
    
    def agent_node(self, state):
        """Agent decides whether to retrieve documents or end."""
        print("---CALL AGENT---")
        messages = state["messages"]
        
        model = ChatOllama(
            model=self.chat_model_name,
            temperature=0,
            base_url=self.ollama_base_url
        ).bind_tools([self.retriever_tool])
        
        response = model.invoke(messages)
        return {"messages": [response]}
    
    def rewrite_node(self, state):
        """Transform the query to produce a better question."""
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content
        
        msg = [
            HumanMessage(
                content=f"""Look at the input and try to reason about the underlying semantic intent.

Initial question:
{question}

Formulate an improved question:""",
            )
        ]
        
        rewrite_model = ChatOllama(
            model=self.chat_model_name,
            temperature=0,
            base_url=self.ollama_base_url
        )
        response = rewrite_model.invoke(msg)
        return {"messages": [response]}
    
    def generate_node(self, state):
        """Generate answer using retrieved documents."""
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        prompt = PromptTemplate(
            template="""You are a helpful assistant for question-answering tasks. Use the retrieved context to answer the question. If you don't know the answer, say so. Keep the answer concise but informative.

Question: {question}
Context: {context}
Answer:""",
            input_variables=["question", "context"]
        )
        
        generate_model = ChatOllama(
            model=self.chat_model_name,
            temperature=0,
            base_url=self.ollama_base_url
        )
        rag_chain = prompt | generate_model | StrOutputParser()
        
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    def build_graph(self):
        """Build the LangGraph workflow."""
        if not self.retriever_tool:
            raise ValueError("Must add documents before building graph")
            
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite", self.rewrite_node)
        workflow.add_node("generate", self.generate_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "retrieve", END: END},
        )
        workflow.add_conditional_edges("retrieve", self.grade_documents)
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        
        self.graph = workflow.compile()
        print("Graph compiled successfully!")
        
    def query(self, question: str, verbose: bool = True) -> str:
        """Run a query through the agentic RAG system."""
        if not self.graph:
            raise ValueError("Must build graph before querying")
            
        inputs = {"messages": [("user", question)]}
        
        if verbose:
            print(f"\n=== Processing Query: {question} ===")
        
        final_response = None
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                if verbose:
                    print(f"\nOutput from node '{key}'")
                if key == "generate" and "messages" in value:
                    final_response = value["messages"][0]
                    
        return final_response if final_response else "I couldn't generate a response."
    
    def chat(self, message: str, chat_history: List = None) -> tuple:
        """
        Chat interface that maintains conversation history.
        
        Args:
            message: User's message
            chat_history: Previous conversation history
            
        Returns:
            tuple: (response, updated_chat_history)
        """
        if chat_history is None:
            chat_history = []
            
        # Add user message to history
        chat_history.append(("user", message))
        
        # Get response
        response = self.query(message, verbose=False)
        
        # Add assistant response to history
        chat_history.append(("assistant", response))
        
        return response, chat_history


# Example usage and testing
def test_basic_functionality():
    """Test the basic RAG functionality with web URLs."""
    print("Testing basic RAG functionality...")
    
    rag_system = AgenticRAG(
        chat_model="llama3.1:8b",
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434"
    )
    
    # Add documents from URLs
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ]
    
    chunks_added = rag_system.add_documents(urls=urls)
    print(f"Added {chunks_added} chunks from {len(urls)} URLs")
    
    # Build graph
    rag_system.build_graph()
    
    # Test queries
    questions = [
        "What are the types of agent memory?",
        "What is prompt engineering?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        print(f"\nQ: {question}")
        print(f"A: {response}")
        print("-" * 50)


if __name__ == "__main__":
    # Test the system
    test_basic_functionality()