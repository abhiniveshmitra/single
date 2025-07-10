import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langgraph.graph import StateGraph, END

# Load environment variables from the .env file
load_dotenv()

# --- Azure Service Configurations ---
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- 1. Define the Graph's State ---
class VdiAnalysisState(TypedDict):
    """
    Represents the state passed between nodes in the VDI analysis graph.
    """
    input_payload: dict
    search_query: str
    search_results: List[dict]
    analysis_report: dict
    error: str

# --- 2. Define Agent Nodes (Functions) ---

def query_vdi_logs(state: VdiAnalysisState) -> VdiAnalysisState:
    """
    Node to query Azure AI Search for VDI logs based on the user_id from the input.
    """
    print("--- Node: Querying VDI Logs ---")
    try:
        user_id = state.get("input_payload", {}).get("user_id")
        if not user_id:
            return {"error": "user_id not found in the input payload."}

        search_credential = AzureKeyCredential(SEARCH_API_KEY)
        search_client = SearchClient(endpoint=SEARCH_ENDPOINT,
                                     index_name=SEARCH_INDEX_NAME,
                                     credential=search_credential)

        filter_query = f"user_id eq '{user_id}'"
        
        results = search_client.search(
            search_text="*",
            filter=filter_query,
            include_total_count=True,
            top=25
        )

        records = [result for result in results]
        if not records:
            return {"error": f"No VDI logs found for user_id: {user_id}"}
        
        return {"search_results": records}

    except Exception as e:
        return {"error": f"Failed to query Azure AI Search: {e}"}

def analyze_with_vdi_expertise(state: VdiAnalysisState) -> VdiAnalysisState:
    """
    Node that uses an LLM with a specialized prompt to analyze the retrieved VDI logs.
    """
    print("--- Node: Analyzing Data with VDI/Citrix Expertise ---")
    try:
        search_results = state.get("search_results")
        problem_description = state.get("input_payload", {}).get("problem_description")
        
        if not search_results:
            return {"error": "No search results available to analyze."}

        llm = AzureChatOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_deployment=OPENAI_DEPLOYMENT_NAME,
            temperature=0.2
        )

        # EXPERT PROMPT: This prompt frames the AI as a Citrix and VDI specialist.
        system_prompt = (
            "You are an expert-level AI analyst with deep, specialized knowledge of Virtual Desktop Infrastructure (VDI), focusing on Citrix Workspace and its related components. "
            "Your task is to analyze raw VDI monitoring logs in the context of a user-reported problem. "
            "Leverage your expertise to find the root cause, paying close attention to Citrix-specific issues like HDX channel problems, profile loading times, policy misconfigurations, or ICA latency. "
            "Your final output MUST be a single, valid JSON object."
        )

        user_prompt = f"""
        Analyze the following user problem and the associated VDI monitoring logs.
        
        User Problem:
        "{problem_description}"

        VDI Log Data:
        {json.dumps(search_results, indent=2)}

        Based on your expert VDI and Citrix knowledge, generate a structured JSON analysis with the following fields:
        - "analysis_summary": A one-sentence technical summary of the primary issue.
        - "key_vdi_metrics": An object containing key metrics found in the logs (e.g., average_cpu_usage, memory_pressure, ica_round_trip_time).
        - "insights": A list of detailed observations. Specifically mention any Citrix-related findings.
        - "recommendations": A list of numbered, actionable steps for a VDI administrator to resolve the issue.
        - "sentiment": A one-word summary of the user experience ("Poor", "Fair", "Good").
        """

        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        report = json.loads(response.content)
        return {"analysis_report": report}

    except Exception as e:
        return {"error": f"Failed to generate analysis from Azure OpenAI: {e}"}


# --- 3. Build the LangGraph Workflow ---

workflow = StateGraph(VdiAnalysisState)

# Add nodes to the graph
workflow.add_node("query_vdi_logs", query_vdi_logs)
workflow.add_node("analyze_with_vdi_expertise", analyze_with_vdi_expertise)

# Define the graph's flow
workflow.set_entry_point("query_vdi_logs")
workflow.add_edge("query_vdi_logs", "analyze_with_vdi_expertise")
workflow.add_edge("analyze_with_vdi_expertise", END)

# Compile the final agent
vdi_agent = workflow.compile()

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    # Sample payload representing the data that triggers the agent
    sample_payload = {
        "user_id": "a13e5e078-aec5-4345-a630-1ea859a1555a",
        "problem_description": "User reports that their Citrix session is frequently freezing and disconnecting, especially when working with large Excel files."
    }

    inputs = {"input_payload": sample_payload}

    print("--- Starting VDI Expert Agent ---")
    final_state = vdi_agent.invoke(inputs)

    print("\n--- Agent Execution Finished ---")
    if final_state.get("error"):
        print("\n--- ERROR ---")
        print(final_state["error"])
    else:
        print("\n--- Final Analysis Report ---")
        print(json.dumps(final_state.get("analysis_report"), indent=2))
