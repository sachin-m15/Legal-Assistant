import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Any
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import the refactored functions from the legal_rag directory
from legal_rag.query_rewriter import rewrite_query
from legal_rag.retrieval import perform_research
from legal_rag.summarizer import summarize_and_reflect, generate_final_analysis

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Ensure API keys are set up
GROK_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY or not GROK_API_KEY:
    st.error(
        "API keys for Tavily and Grok are not set. Please add them to your .env file."
    )
    st.stop()


# --- State Management with LangGraph ---
class AgentState(TypedDict):
    """
    Represents the state of our legal research agent.
    Each key is a node in the LangGraph.
    """

    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    research_complete: bool

    # Store the conversation history for a multi-turn chat experience
    chat_history: List[BaseMessage]


# --- Build the LangGraph ---


def decide_next_step(state: AgentState):
    """Decides whether to continue research or finalize the analysis."""
    print("---DECIDING NEXT STEP---")
    if state.get("research_complete", False):
        print("---RESEARCH COMPLETE. PROCEEDING TO FINAL ANALYSIS.---")
        return "final_analysis"
    else:
        print("---RESEARCH INCOMPLETE. REWRITING QUERY AND RESEARCHING AGAIN.---")
        return "perform_research"


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("perform_research", perform_research)
workflow.add_node("summarize_and_reflect", summarize_and_reflect)
workflow.add_node("final_analysis", generate_final_analysis)

# Set the entry point
workflow.set_entry_point("rewrite_query")

# Add edges and conditional edges
workflow.add_edge("rewrite_query", "perform_research")
workflow.add_edge("perform_research", "summarize_and_reflect")
workflow.add_conditional_edges(
    "summarize_and_reflect",
    decide_next_step,
    {
        "perform_research": "perform_research",  # Loop back to research
        "final_analysis": "final_analysis",
    },
)
workflow.add_edge("final_analysis", END)

# Compile the graph
app = workflow.compile()

# --- Streamlit UI ---


def main():
    st.title("Legal Research and Analysis Assistant (LARA)")
    st.markdown(
        "Enter a legal problem or case description below to get a comprehensive analysis based on Indian law."
    )

    user_query = st.text_area(
        "Your legal query:",
        placeholder="e.g., 'What are the legal provisions for noise pollution in residential areas in India?'",
        height=150,
    )

    if st.button("Get Legal Analysis"):
        if not user_query:
            st.warning("Please enter a query to proceed.")
            return

        with st.spinner("Conducting legal research..."):
            try:
                # Initial state
                initial_state = {
                    "query": user_query,
                    "intermediate_steps": [],
                    "chat_history": [HumanMessage(content=user_query)],
                }

                # Run the graph
                for s in app.stream(initial_state):
                    print(s)  # For debugging

                # Retrieve the final state
                final_state = app.invoke(initial_state)

                # Display the result
                st.subheader("Legal Analysis")
                st.markdown(final_state["final_analysis"])

                st.subheader("Research Process (Intermediate Steps)")
                for step in final_state["intermediate_steps"]:
                    st.text(step)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(
                    "Please ensure you have a pre-indexed FAISS vector store and valid API keys."
                )


if __name__ == "__main__":
    main()
