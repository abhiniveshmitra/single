import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

import logging

# Load environment variables from .env
load_dotenv()

# Get environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
embedding_deployment = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT")
llm_deployment = os.getenv("OPENAI_LLM_DEPLOYMENT")

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
azure_search_index = os.getenv("AZURE_SEARCH_INDEX")

# Embedding function (correct deployment name!)
embedding_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment=embedding_deployment,  # use env var
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base
)

# LLM for chat (correct deployment name!)
llm = ChatOpenAI(
    azure_deployment=llm_deployment,  # use env var
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base
)

# Azure Search vectorstore
vectorstore = AzureSearch(
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_key,
    index_name=azure_search_index,
    embedding_function=embedding_function,
)

retriever = vectorstore.as_retriever()

# ---- Your workflow ----

def search_cqd(state):
    user_id = state["user_id"]
    docs = retriever.get_relevant_documents(query=user_id)
    return {"user_id": user_id, "docs": docs}

def query_openai(state):
    docs = state["docs"]
    prompt = f"""
    You are a Microsoft Teams call diagnostics assistant.

    A user recently had a poor-quality Teams call. Based on telemetry data from recent and past calls given below

    {docs}

    Your output must:
    1. Clearly summarize the root cause in plain English ('issue_summary')
    2. Select the most appropriate actionable insight from this list:
       1. Automate clearing its cache and relaunching the app...
       ...
       5. Automate resolving screen sharing issues...
    Respond in JSON:
    {{
        "summary": "...",
        "insight": "..."
    }}
    """
    response = llm.predict(prompt)
    return {**state, "result": response}

def save_to_cosmos(state):
    logging.info("Saved to cosmos")
    # store_insight(state["user_id"], state["summary"], state["insight"])
    return state

graph = StateGraph()
graph.add_node("search", search_cqd)
graph.add_node("reason", query_openai)
graph.add_node("store", save_to_cosmos)

graph.set_entry_point("search")
graph.add_edge("search", "reason")
graph.add_edge("reason", "store")
graph.add_edge("store", "search")

langgraph_app = graph.compile()

