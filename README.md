# L.A.R.A - Legal Analysis Research Assistant

## Overview

L.A.R.A is an intelligent, Python-based application designed to assist legal professionals and citizens in conducting comprehensive research and case studies based on Indian law. Inspired by the principles of deep research agents, this tool automates the legal analysis workflow by transforming user queries into precise search terms, retrieving information from multiple sources, and synthesizing a cohesive, citable legal analysis.

The system is built using a modular state graph orchestrated by **LangGraph** to ensure a scalable and maintainable architecture.

## Features

* **Intelligent Query Rewriting:** Automatically reformulates legal problems or case descriptions into optimized, legally precise search queries tailored to the user's role.
* **Role-Based Agents:** Offers two specialized workflows for **Common Citizens** (for general legal guidance) and **Lawyers** (for in-depth case analysis).
* **Hybrid Data Retrieval:** Queries a pre-indexed FAISS vector store containing a vast corpus of Indian laws and uses a web search API for supplementary information and recent developments.
* **Iterative Research:** The system dynamically evaluates retrieved information, identifies knowledge gaps, and generates new search queries in a continuous loop until a comprehensive analysis is achieved.
* **Cohesive Legal Analysis:** Combines findings from both internal documents and web sources into a single, structured report.
* **Citations and Recommendations:** The final output includes clear citations to relevant legal documents and web pages, along with a summary of key findings and actionable recommendations.

## Tech Stack

* **Framework:** LangGraph (for agent orchestration), Streamlit (for the UI)
* **Language Models:** Groq (Llama-3.1-8b-instant)
* **RAG:** FAISS (for vector search), Tavily Search API (for web search)
* **Development:** Python, `dotenv`

## Project Structure
```
LARA/
├── app.py                      # Main Streamlit app - entry point for the UI
│
├── agent/                      # AI agents & workflows
│   ├── init.py             # Makes the 'agent' directory a Python package
│   ├── citizen_agent.py        # Agent workflow for citizens (legal awareness, guidance)
│   ├── lawyer_agent.py         # Agent workflow for lawyers (case insights, references)
│   └── router.py               # Routes requests to the right agent based on user role
│
├── models_score_checker.py     # Model evaluation + performance visualization in Streamlit
│
├── legal_rag/                  # Retrieval-Augmented Generation (RAG) modules
│   ├── retrieval.py            # FAISS-based retrieval of law documents
│   ├── query_rewriter.py       # Rewrites queries for better search results
│   └── summarizer.py           # Summarizes results, lawyer-specific reports
│
├── data/                       # Indexed & raw data
│   ├── faiss_index/            # Vector DB (FAISS index) for retrieval
│   ├── indian_law_docs/        # Raw Indian legal documents
│   └── data_converter.py       # Converts PDFs/JSON into text for indexing
│
├── raw_data/                   # Original PDFs & JSON files (source legal docs)
│
├── requirements.txt            # Dependencies list (install with pip)
└── README.md                   # Documentation
```


## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sachin-m15/L.A.R.A-Legal-Analysis-Research-Assistant-.git](https://github.com/sachin-m15/L.A.R.A-Legal-Analysis-Research-Assistant-.git)
    cd L.A.R.A-Legal-Analysis-Research-Assistant-
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables:**
    Create a `.env` file in the root directory and add your API keys. You can obtain these from their respective websites.
    ```
    GROQ_API_KEY="your_grok_api_key_here"
    TAVILY_API_KEY="your_tavily_api_key_here"
    ```

5.  **Prepare your Legal Data:**
    * Place your legal documents (PDFs, JSONs) in the `raw_data/` directory.
    * Run the `data_converter.py` script to process them and create the FAISS index.
    ```bash
    python data/data_converter.py
    ```

## How to Run

After installation and setup, run the application from the root directory:

```bash
streamlit run app.py
