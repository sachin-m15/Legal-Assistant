import os
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableParallel
from typing import TypedDict, Annotated, List, Any
import operator
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# Agent State
# -------------------------
class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str


# -------------------------
# FAISS Legal DB Tool
# -------------------------
@tool
def legal_database_search(query: str) -> str:
    """
    Search against a pre-indexed FAISS vector store of Indian laws and cases.
    """
    try:
        FAISS_INDEX_PATH = "data/faiss_index"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
        )

        retrieved_docs = db.similarity_search(query, k=5)
        return " ".join([doc.page_content for doc in retrieved_docs])

    except FileNotFoundError:
        return f"FAISS index not found at {FAISS_INDEX_PATH}. Please pre-index your legal data."
    except Exception as e:
        return f"Error during legal database search: {e}"


# -------------------------
# Research Function
# -------------------------
def perform_research(state: AgentState) -> dict:
    """Performs both FAISS and web searches in parallel."""
    print("---PERFORMING RESEARCH---")
    query = state["query"]

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    # The variable for the tool is correctly named 'web_search_tool' here
    web_search_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)

    # Run both searches in parallel
    rag_chain = RunnableParallel(
        {
            "faiss_search_results": lambda x: legal_database_search.invoke(x["query"]),
            # Change 'tavily_search_tool' to 'web_search_tool' to match the variable name
            "web_search_results": lambda x: web_search_tool.invoke(x["query"]),
        }
    )

    # Pass {"query": query} instead of full state
    results = rag_chain.invoke({"query": query})
    print("RAW RESULTS:", results)

    # Normalize web search results (list of dicts → string)
    web_results = results.get("web_search_results", "")
    if isinstance(web_results, list):
        web_results_str = " ".join([r.get("content", str(r)) for r in web_results if r])
    else:
        web_results_str = str(web_results)

    return {
        "faiss_search_results": results.get("faiss_search_results", ""),
        "web_search_results": web_results_str,
        "intermediate_steps": [
            f"FAISS Results: {results.get('faiss_search_results', '')}",
            f"Web Results: {web_results_str}",
        ],
    }
