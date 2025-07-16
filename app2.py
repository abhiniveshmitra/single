import os
import uuid
import logging
import pandas as pd
import io
from contextlib import redirect_stdout
from typing import TypedDict, List, Any, Dict
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Graph State Definition ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        keys: A dictionary to hold session data.
    """
    keys: Dict[str, Any]

# --- Client Setup ---
def create_search_client() -> SearchClient:
    """Creates and returns an Azure AI Search client."""
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX") # Confirmed as "teams" from your image [1]
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))

def create_openai_client() -> AzureChatOpenAI:
    """Creates and returns an Azure OpenAI client."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )

# --- LangGraph Nodes ---

def analysis_node(state: GraphState) -> GraphState:
    """
    Fetches data, prepares it, and generates an analysis script.
    """
    logger.info("---ANALYSIS NODE---")
    state_keys = state['keys']
    question = state_keys['question']
    search_client = state_keys['search_client']
    llm = state_keys['llm']

    # 1. Fetch all data directly from the Azure Search Index
    logger.info("Fetching all documents from the index...")
    try:
        # Using search="*" retrieves all documents. Note: Azure Search has a default limit of 1000 docs per query.
        # For very large datasets, pagination would be required.
        search_results = search_client.search(search_text="*", top=5000, select=["*"])
        documents = [result for result in search_results]
        if not documents:
            logger.warning("No documents found in the index. Stopping analysis.")
            state_keys['final_answer'] = "No data was found in the search index to analyze."
            state_keys['generated_code'] = None
            return {"keys": state_keys}
        logger.info(f"Successfully fetched {len(documents)} documents.")
    except Exception as e:
        logger.error(f"Failed to fetch documents from Azure Search: {e}")
        state_keys['final_answer'] = f"Error: Could not retrieve data from the index. {e}"
        state_keys['generated_code'] = None
        return {"keys": state_keys}

    # 2. Load data into a pandas DataFrame and clean it
    df = pd.DataFrame(documents)
    
    # Clean numeric columns to ensure they can be used in calculations
    numeric_cols = [
        "callDetails_Score", "audioQuality_Score", "videoQuality_Score",
        "sessions_segments_media_streams_averageAudioNetworkJitter",
        "sessions_segments_media_streams_averageVideoPacketLossRate",
        "sessions_segments_media_streams_averageVideoFrameRate"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Store the DataFrame for the execution node
    state_keys['dataframe'] = df

    # 3. Generate a Python script based on the available data
    system_prompt = """
    You are a master VDI data analyst. A pandas DataFrame named `df` has been pre-loaded and is available in your environment.
    Your task is to write a Python script to analyze this DataFrame to answer the user's question.

    Instructions:
    - Your script must use the DataFrame named `df`.
    - Do not include code to load data (e.g., from a CSV).
    - Analyze the data and use the `print()` function to output your findings in a clear, human-readable format.
    - The output should be the final answer, not just raw data.
    """

    prompt = f"""
    User Question: "{question}"

    Here is the head of the available DataFrame `df`:
    {df.head().to_markdown()}

    Write the Python code to perform the analysis and print the answer now.
    """

    response = llm.invoke(system_prompt + prompt)
    # The generated code is Python code block, extract it
    generated_code = response.content.strip().replace("``````", "")
    state_keys['generated_code'] = generated_code
    
    return {"keys": state_keys}

def execution_node(state: GraphState) -> GraphState:
    """
    Executes the generated Python code to get the final answer.
    """
    logger.info("---EXECUTION NODE---")
    state_keys = state['keys']
    generated_code = state_keys.get('generated_code')

    if not generated_code:
        logger.warning("No code was generated, skipping execution.")
        return {"keys": state_keys}

    df = state_keys['dataframe']
    
    # Prepare the execution environment
    local_scope = {"df": df, "pd": pd}
    output_buffer = io.StringIO()

    logger.info("Executing generated code...")
    try:
        # Capture the print output from the executed code
        with redirect_stdout(output_buffer):
            exec(generated_code, {"pd": pd}, local_scope)
        
        final_answer = output_buffer.getvalue()
        state_keys['final_answer'] = final_answer
        logger.info("Execution successful.")
    except Exception as e:
        logger.error(f"Error executing generated code: {e}")
        state_keys['final_answer'] = f"An error occurred during analysis: {e}"

    return {"keys": state_keys}

# --- Graph Construction ---
def build_graph() -> StateGraph:
    """Builds and returns the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    workflow.add_node("analyst", analysis_node)
    workflow.add_node("executor", execution_node)

    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "executor")
    workflow.add_edge("executor", END)
    
    return workflow.compile()

# --- Main Execution ---
if __name__ == "__main__":
    app = build_graph()

    # The agent now uses its internal clients, so we don't pass them in the initial state
    initial_state = {
        "keys": {
            "question": "Which devices are associated with the worst video quality scores and high audio jitter? Show me the top 5.",
            "search_client": create_search_client(),
            "llm": create_openai_client(),
        }
    }

    logger.info(f"Invoking VDI Analyst Agent with question: '{initial_state['keys']['question']}'")
    final_state = app.invoke(initial_state)

    print("\n-------------------- FINAL ANSWER --------------------\n")
    print(final_state['keys'].get('final_answer', 'No answer was produced.'))
    print("\n------------------------------------------------------\n")
