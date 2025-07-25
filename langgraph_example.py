"""
Simple LangGraph-based Agentic RAG Application using Ollama Models
Based on: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/

This implementation uses Ollama instead of OpenAI models and is designed to be modular for easy extension.
"""

import os
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

# Core LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool

# Ollama and community imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
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
            ollama_base_url: Base URL for Ollama server (default: http://localhost:11434)
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
        
    def setup_retriever(self, urls: list[str]):
        """
        Set up the document retriever from web URLs.
        
        Args:
            urls: List of URLs to load and index
        """
        print("Loading documents...")
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=self.embeddings,
        )
        
        self.retriever = vectorstore.as_retriever()
        
        # Create retriever tool
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_blog_posts",
            "Search and return information about blog posts on LLM agents, prompt engineering, and adversarial attacks.",
        )
        
        print(f"Indexed {len(doc_splits)} document chunks")
        
    def grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.
        
        Args:
            state: The current state containing messages
            
        Returns:
            str: Decision for whether documents are relevant ("generate") or not ("rewrite")
        """
        print("---CHECK RELEVANCE---")
        
        # Get the question and retrieved documents
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        # Create grading chain with base URL
        grading_model = ChatOllama(
            model=self.chat_model_name,
            temperature=0,
            base_url=self.ollama_base_url
        )
        llm_with_tool = grading_model.with_structured_output(DocumentRelevanceGrade)
        
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question.

Here is the retrieved document:
{context}

Here is the user question: {question}

If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
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
        """
        Agent node that decides whether to retrieve documents or end.
        
        Args:
            state: The current state containing messages
            
        Returns:
            dict: Updated state with agent response
        """
        print("---CALL AGENT---")
        messages = state["messages"]
        
        # Bind tools to model with base URL
        model = ChatOllama(
            model=self.chat_model_name,
            temperature=0,
            base_url=self.ollama_base_url
        ).bind_tools([self.retriever_tool])
        response = model.invoke(messages)
        
        return {"messages": [response]}
    
    def rewrite_node(self, state):
        """
        Transform the query to produce a better question.
        
        Args:
            state: The current state containing messages
            
        Returns:
            dict: Updated state with re-phrased question
        """
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content
        
        msg = [
            HumanMessage(
                content=f"""Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:
------- 
{question}
------- 
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
        """
        Generate answer using retrieved documents.
        
        Args:
            state: The current state containing messages
            
        Returns:
            dict: Updated state with generated answer
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        # Create RAG prompt
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:""",
            input_variables=["question", "context"]
        )
        
        # Create RAG chain with base URL
        generate_model = ChatOllama(
            model=self.chat_model_name,
            temperature=0,
            base_url=self.ollama_base_url
        )
        rag_chain = prompt | generate_model | StrOutputParser()
        
        # Generate response
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    def build_graph(self):
        """Build the LangGraph workflow."""
        if not self.retriever_tool:
            raise ValueError("Must call setup_retriever() before building graph")
            
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite", self.rewrite_node)
        workflow.add_node("generate", self.generate_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        
        # Conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # Conditional edges from retrieve
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
        )
        
        # Final edges
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        
        # Compile graph
        self.graph = workflow.compile()
        print("Graph compiled successfully!")
        
    def query(self, question: str):
        """
        Run a query through the agentic RAG system.
        
        Args:
            question: The user's question
            
        Returns:
            dict: The final response from the system
        """
        if not self.graph:
            raise ValueError("Must call build_graph() before querying")
            
        inputs = {"messages": [("user", question)]}
        
        print(f"\n=== Processing Query: {question} ===")
        
        final_response = None
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print(f"\nOutput from node '{key}':")
                if key == "generate" and "messages" in value:
                    final_response = value["messages"][0]
                    print(f"Generated Answer: {final_response}")
                    
        return final_response


def main():
    """Example usage of the Agentic RAG system."""
    
    # Initialize the system with local Ollama server
    rag_system = AgenticRAG(
        chat_model="llama3.1:8b",  # Make sure this model is available in Ollama
        embedding_model="nomic-embed-text",  # Make sure this model is available in Ollama
        ollama_base_url="http://localhost:11434"  # Default Ollama server URL
    )
    
    # Set up retriever with blog posts
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    rag_system.setup_retriever(urls)
    
    # Build the graph
    rag_system.build_graph()
    
    # Example queries
    questions = [
        "What does Lilian Weng say about the types of agent memory?",
        "What are the main components of an LLM agent?",
        "How do adversarial attacks work on language models?"
    ]
    
    for question in questions:
        response = rag_system.query(question)
        print(f"\nFinal Answer: {response}")
        print("=" * 80)


if __name__ == "__main__":
    # Make sure you have Ollama running locally and the required models installed:
    # 1. Start Ollama server: ollama serve (if not running as service)
    # 2. Pull required models:
    #    ollama pull llama3.1:8b
    #    ollama pull nomic-embed-text
    # 3. Verify models are available: ollama list
    
    main()