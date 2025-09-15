import os
import operator
import asyncio  # noqa: F401
from typing import TypedDict, Annotated, List, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage  # noqa: F401

load_dotenv()

# ------------------------------
# Config
# ------------------------------
FAST_MODE = True  # ✅ Toggle True = faster (trims), False = detailed chunking
CHUNK_SIZE = 1200
MAX_CHUNKS = 3


# ------------------------------
# Agent State
# ------------------------------
class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    research_complete: bool
    chat_history: List[BaseMessage]


# ------------------------------
# Utility Functions
# ------------------------------
def get_llm():
    """Initialize Groq LLM with env key."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=groq_api_key
    )


def safe_invoke(llm, prompt, vars):
    """Run a prompt safely and return text content."""
    chain = prompt | llm
    result = chain.invoke(vars)
    return getattr(result, "content", str(result))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, max_chunks: int = MAX_CHUNKS):
    """Split text into word chunks, capped to max_chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if len(chunks) >= max_chunks:
            break
    return chunks


def summarize_long_text(text: str, label: str, query: str) -> str:
    """Summarize text with fast or detailed strategy."""
    if not text:
        return f"No {label} found."

    llm = get_llm()

    # ✅ Fast mode: just trim input
    if FAST_MODE:
        trimmed = " ".join(text.split()[:2000])
        prompt = PromptTemplate(
            template=f"""Summarize the following {label} (<200 words), focusing on acts, sections, judgments.

            Query: {{query}}
            Text: {{text}}

            Summary:""",
            input_variables=["query", "text"],
        )
        return safe_invoke(llm, prompt, {"query": query, "text": trimmed})

    # ✅ Detailed mode: chunk + merge
    chunk_summaries = []
    for chunk in chunk_text(text):
        prompt = PromptTemplate(
            template=f"""Summarize this {label} chunk (<120 words),
            focusing only on acts, sections, judgments.

            Query: {{query}}
            {label} chunk: {{chunk}}

            Summary:""",
            input_variables=["query", "chunk"],
        )
        chunk_summaries.append(
            safe_invoke(llm, prompt, {"query": query, "chunk": chunk})
        )

    merge_prompt = PromptTemplate(
        template=f"""Combine the following {label} summaries into one concise digest (<250 words):

        Query: {{query}}
        Summaries:
        {{summaries}}

        Final Digest:""",
        input_variables=["query", "summaries"],
    )
    return safe_invoke(
        llm, merge_prompt, {"query": query, "summaries": "\n".join(chunk_summaries)}
    )


# ------------------------------
# Core Functions
# ------------------------------
def summarize_and_reflect(state: AgentState) -> dict:
    """Summarizes the findings and reflects on the research to identify gaps."""
    print("---SUMMARIZING & REFLECTING---")
    query = state["query"]

    faiss_summary = summarize_long_text(
        state["faiss_search_results"], "FAISS results", query
    )
    web_summary = summarize_long_text(state["web_search_results"], "Web results", query)

    llm = get_llm()
    summary_prompt = PromptTemplate(
        template="""Based on the original query and the following summarized search results,
        provide a concise reflection.

        Original Query: {query}
        FAISS Summary: {faiss_summary}
        Web Summary: {web_summary}

        Reflection:
        1. Key findings:
        2. Knowledge Gaps:
        3. Research complete? ('YES' or 'NO')""",
        input_variables=["query", "faiss_summary", "web_summary"],
    )

    summary = safe_invoke(
        llm,
        summary_prompt,
        {"query": query, "faiss_summary": faiss_summary, "web_summary": web_summary},
    )

    is_complete = "YES" in summary.upper()

    return {
        "intermediate_steps": state["intermediate_steps"] + [summary],
        "research_complete": is_complete,
    }


def generate_final_analysis(state: AgentState) -> dict:
    """Generates the final, structured legal analysis."""
    print("---GENERATING FINAL ANALYSIS---")
    query = state["query"]
    all_steps = "\n".join(state["intermediate_steps"])

    # Compress steps if too long
    if len(all_steps.split()) > 1500:
        all_steps = summarize_long_text(all_steps, "research steps", query)

    llm = get_llm()
    analysis_prompt = PromptTemplate(
        template="""Based on the user query and research steps,
        generate a structured legal analysis:

        - **Original Query**
        - **Legal Context**
        - **Case Law Summary**
        - **Analysis and Recommendations**
        - **Sources**

        Query: {query}
        Research Steps: {all_steps}

        Final Analysis:""",
        input_variables=["query", "all_steps"],
    )

    final_analysis = safe_invoke(
        llm, analysis_prompt, {"query": query, "all_steps": all_steps}
    )

    return {"final_analysis": final_analysis}
