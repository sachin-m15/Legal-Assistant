import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Annotated, List, Any
import operator
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str


def rewrite_query(state: AgentState) -> dict:
    """
    Rewrites the user's natural language query into a precise legal search query.
    """
    print("---REWRITING QUERY---")
    query = state["query"]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # use a supported model
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # Prompt instructs the model to return JSON
    rewrite_prompt = PromptTemplate(
        template="""
You are a legal research assistant specializing in Indian law.
Rewrite the user's query into a precise, legally-focused search query.

Output MUST be JSON in this format:
{{ "rewritten_query": "<your query here>" }}

Original Query: {query}
JSON Output:
""",
        input_variables=["query"],
    )

    try:
        rewriter_chain = rewrite_prompt | llm | JsonOutputParser()
        rewritten_query = rewriter_chain.invoke({"query": query})
    except Exception as e:
        print(f"JSON parsing failed, using raw string. Error: {e}")
        # fallback: just return the raw rewritten text
        rewritten_text = llm.invoke({"input": query})
        rewritten_query = {"rewritten_query": rewritten_text}

    print(f"Rewritten Query: {rewritten_query}")
    return rewritten_query
