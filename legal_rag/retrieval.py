# LARA/legal_rag/retrieval.py

import os
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableParallel
from langchain_core.documents import Document  # <-- NEW: Import Document
from langchain_core.messages import BaseMessage  # <-- FIX: Import BaseMessage
from typing import TypedDict, Annotated, List, Any
import operator
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# Agent State (updated to match new structure)
# -------------------------
class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    research_complete: bool
    chat_history: List[BaseMessage]
    sources: Annotated[List[dict], operator.add]  # <-- NEW: To store source metadata


# -------------------------
# FAISS Legal DB Tool (Updated to return Document objects)
# -------------------------
@tool
def legal_database_search(query: str) -> List[Document]:
    """
    Search against a pre-indexed FAISS vector store of Indian laws and cases.
    Returns a list of Document objects with page content and metadata.
    """
    try:
        FAISS_INDEX_PATH = "data/faiss_index"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
        )

        retrieved_docs = db.similarity_search_with_score(query, k=5)

        # You'll need to ensure your FAISS index stores metadata for each document,
        # such as the file name, case name, or source.

        return [doc[0] for doc in retrieved_docs]  # Return list of Document objects

    except FileNotFoundError:
        return [Document(page_content=f"FAISS index not found at {FAISS_INDEX_PATH}.")]
    except Exception as e:
        return [Document(page_content=f"Error during legal database search: {e}")]


# -------------------------
# Research Function (Updated to handle structured output and sources)
# -------------------------
def perform_research(state: AgentState) -> dict:
    """Performs both FAISS and web searches in parallel."""
    print("---PERFORMING RESEARCH---")
    query = state["query"]

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    web_search_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)

    rag_chain = RunnableParallel(
        {
            "faiss_search_results": lambda x: legal_database_search.invoke(x["query"]),
            "web_search_results": lambda x: web_search_tool.invoke(x["query"]),
        }
    )

    results = rag_chain.invoke({"query": query})

    faiss_docs = results.get("faiss_search_results", [])
    web_results_list = results.get("web_search_results", [])

    # Extract source metadata and create a single list of sources
    sources = []

    # Process FAISS sources
    faiss_content = ""
    for doc in faiss_docs:
        faiss_content += doc.page_content + "\n\n"
        if doc.metadata:
            sources.append({"type": "document", "metadata": doc.metadata})

    # Process web sources
    web_content = ""
    for item in web_results_list:
        web_content += item.get("content", "") + "\n\n"
        sources.append(
            {
                "type": "web",
                "url": item.get("url", "N/A"),
                "title": item.get("title", "N/A"),
            }
        )

    print("RAW RESULTS:", {"faiss": faiss_content, "web": web_content})

    return {
        "faiss_search_results": faiss_content,
        "web_search_results": web_content,
        "sources": sources,
        "intermediate_steps": [
            f"FAISS Results: {faiss_content}",
            f"Web Results: {web_content}",
        ],
    }
