import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Any
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # ✅ use memory saver

# Import the refactored functions from the legal_rag directory
from legal_rag.query_rewriter import rewrite_query
from legal_rag.retrieval import perform_research
from legal_rag.summarizer import summarize_and_reflect, generate_final_analysis

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GROK_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY or not GROK_API_KEY:
    st.error(
        "API keys for Tavily and Grok are not set. Please add them to your .env file."
    )
    st.stop()


# --- State Management with LangGraph ---
class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    research_complete: bool
    chat_history: List[BaseMessage]  # conversation history


def decide_next_step(state: AgentState):
    """Decide whether to continue research or finalize the analysis."""
    if state.get("research_complete", False):
        return "final_analysis"
    else:
        return "perform_research"


# --- Build the LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("perform_research", perform_research)
workflow.add_node("summarize_and_reflect", summarize_and_reflect)
workflow.add_node("final_analysis", generate_final_analysis)

workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "perform_research")
workflow.add_edge("perform_research", "summarize_and_reflect")
workflow.add_conditional_edges(
    "summarize_and_reflect",
    decide_next_step,
    {
        "perform_research": "perform_research",
        "final_analysis": "final_analysis",
    },
)
workflow.add_edge("final_analysis", END)

# ✅ Use in-memory checkpointing
checkpointer = MemorySaver()

# Compile the graph
app = workflow.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    """With MemorySaver, we can only retrieve threads from memory (non-persistent)."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
