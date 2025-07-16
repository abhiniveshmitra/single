import os
import io
import logging
import pandas as pd
from contextlib import redirect_stdout
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables
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
    return SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )

def create_openai_client() -> AzureChatOpenAI:
    """Creates and returns an Azure OpenAI client."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.1 # Slightly increased for more nuanced text generation
    )

# --- LangGraph Nodes ---

def analysis_node(state: GraphState) -> GraphState:
    """
    Fetches data from the index and generates Python code to extract raw facts.
    """
    logger.info("---ANALYSIS NODE: Generating data extraction script---")
    state_keys = state['keys']
    question = state_keys['question']
    search_client = state_keys['search_client']
    llm = state_keys['llm']

    logger.info("Fetching all documents for analysis...")
    try:
        search_results = search_client.search(search_text="*", top=5000, select=["*"])
        documents = [result for result in search_results]
        if not documents:
            state_keys['final_answer'] = "No data was found in the search index to analyze."
            state_keys['generated_code'] = None
            return {"keys": state_keys}
        logger.info(f"Successfully fetched {len(documents)} documents.")
    except Exception as e:
        logger.error(f"Failed to fetch documents from Azure Search: {e}")
        state_keys['final_answer'] = f"Error: Could not retrieve data from the index. {e}"
        state_keys['generated_code'] = None
        return {"keys": state_keys}

    df = pd.DataFrame(documents)
    numeric_cols = [
        "callDetails_Score", "audioQuality_Score", "videoQuality_Score",
        "sessions_segments_media_streams_averageAudioNetworkJitter",
        "sessions_segments_media_streams_averageVideoPacketLossRate"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    state_keys['dataframe'] = df

    system_prompt = """
    You are a data extraction bot. A pandas DataFrame named `df` is available.
    Your task is to write a Python script to filter this DataFrame based on the user's question and print the raw results.
    The output should be just the data, not an explanation. Print the resulting DataFrame as a markdown string. This output will be used by another AI.
    """

    prompt = f"""
    User Question: "{question}"
    Available DataFrame columns: {df.columns.to_list()}

    Write the Python code to extract the relevant data and print it as markdown.
    """
    response = llm.invoke(system_prompt + prompt)
    generated_code = response.content.strip().replace("``````", "")
    state_keys['generated_code'] = generated_code
    
    return {"keys": state_keys}

def execution_node(state: GraphState) -> GraphState:
    """
    Executes the generated Python code to get the raw data facts.
    """
    logger.info("---EXECUTION NODE: Extracting raw data---")
    state_keys = state['keys']
    generated_code = state_keys.get('generated_code')
    if not generated_code:
        logger.warning("No code generated, skipping execution.")
        return {"keys": state_keys}

    df = state_keys['dataframe']
    output_buffer = io.StringIO()
    logger.info("Executing generated code...")
    try:
        with redirect_stdout(output_buffer):
            exec(generated_code, {"pd": pd, "df": df})
        
        # This is the raw data output, not the final answer
        state_keys['analysis_output'] = output_buffer.getvalue()
        logger.info("Data extraction successful.")
    except Exception as e:
        logger.error(f"Error executing generated code: {e}")
        state_keys['analysis_output'] = f"An error occurred during data extraction: {e}"

    return {"keys": state_keys}

def synthesis_node(state: GraphState) -> GraphState:
    """
    Takes the raw data and synthesizes expert, actionable VDI insights.
    """
    logger.info("---SYNTHESIS NODE: Generating expert insights---")
    state_keys = state['keys']
    analysis_output = state_keys.get('analysis_output')
    if not analysis_output or "error" in analysis_output.lower():
        state_keys['final_answer'] = "Could not generate insights due to an error in the data extraction phase."
        return {"keys": state_keys}

    llm = state_keys['llm']
    question = state_keys['question']

    system_prompt = """
    You are a senior VDI performance analyst, a world-class expert in Microsoft Azure Virtual Desktop (AVD) and Citrix Workspace.
    Your task is to transform raw analytical data into actionable, expert-level insights for an IT administrator.
    Do not show any code, DataFrames, or raw numbers. Your response must be pure, professional consultation.
    Think like an IT architect. Structure your response with a summary, key findings, and specific recommendations.
    """

    prompt = f"""
    The user's original request was: "{question}"

    My data extraction script produced the following raw data identifying the problem areas:
    --- RAW DATA ---
    {analysis_output}
    --- END RAW DATA ---

    Now, please provide a concise, expert analysis. Your report should include:
    1.  **Executive Summary:** A brief overview of the findings.
    2.  **Key Findings:** A bulleted list detailing the specific issues identified in the data (e.g., which users, devices, or call types are affected).
    3.  **Actionable VDI Recommendations:** A bulleted list of concrete steps the VDI administrator should take. Be specific to Citrix or AVD where possible. For example, mention Group Policy Objects (GPOs), network optimization (like Citrix HDX or AVD RDP Shortpath), endpoint device health checks, and application tuning.
    """
    
    response = llm.invoke(system_prompt + prompt)
    state_keys['final_answer'] = response.content
    logger.info("Expert insights generated successfully.")
    
    return {"keys": state_keys}

# --- Graph Construction ---
def build_graph() -> StateGraph:
    """Builds and returns the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    workflow.add_node("analyst", analysis_node)
    workflow.add_node("executor", execution_node)
    workflow.add_node("synthesizer", synthesis_node)

    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

# --- Main Execution ---
if __name__ == "__main__":
    app = build_graph()

    initial_state = {
        "keys": {
            "question": "Identify users with poor Teams call quality (low video score, high jitter) and suggest VDI-specific optimizations.",
            "search_client": create_search_client(),
            "llm": create_openai_client(),
        }
    }

    logger.info(f"Invoking VDI Expert Analyst with request: '{initial_state['keys']['question']}'")
    final_state = app.invoke(initial_state)

    print("\n\n================ VDI PERFORMANCE ANALYSIS REPORT ================\n")
    print(final_state['keys'].get('final_answer', 'No answer was produced.'))
    print("\n======================= END OF REPORT =========================\n")

