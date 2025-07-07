import os
import json
import time  # Import the time module for profiling
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Configuration for Azure Services (loaded from .env file) ---
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def get_call_analysis(organizer_user_id: str, participant_user_id: str) -> dict:
    """
    Queries the 'teams-calls' index with corrected field names, profiles the
    API calls, and uses an LLM to generate a detailed JSON analysis.

    Args:
        organizer_user_id: The user ID of the call organizer.
        participant_user_id: The user ID of the call participant to analyze.

    Returns:
        A dictionary containing the AI-generated analysis and profiling data.
    """
    # 1. --- Query Azure AI Search with Profiling ---
    try:
        search_credential = AzureKeyCredential(SEARCH_API_KEY)
        search_client = SearchClient(endpoint=SEARCH_ENDPOINT,
                                     index_name=SEARCH_INDEX_NAME,
                                     credential=search_credential)

        filter_query = (
            f"organizer_user_id eq '{organizer_user_id}' and "
            f"participants_user_id eq '{participant_user_id}'"
        )
        
        select_fields = [
            "organizer_user_id", "organizer_user_displayName", "organizer_device",
            "participants_user_id", "participants_user_displayName", "participants_device",
            "startDateTime", "endDateTime",
            "sessions_segments_media_streams_averagePacketLossRate",
            "sessions_segments_media_streams_averageJitter",
            "sessions_segments_media_streams_averageRoundTripTime"
        ]

        # Start the timer just before the search call
        start_time_search = time.perf_counter()
        
        results = search_client.search(
            search_text="*",
            filter=filter_query,
            select=",".join(select_fields),
            include_total_count=True,
            top=50
        )
        call_records = [result for result in results]
        
        # Stop the timer immediately after the call completes
        end_time_search = time.perf_counter()
        search_duration = end_time_search - start_time_search

        if not call_records:
            return {"error": "No call records found for the specified users."}

    except Exception as e:
        return {"error": f"Failed to query Azure AI Search: {e}"}

    # 2. --- Craft an Intelligent Prompt Aware of the Schema ---
    system_prompt = (
        "You are a meticulous AI data analyst for Microsoft Teams. Your task is to process a JSON array of call records with complex, nested field names. "
        "You must perform calculations on the provided data and present a factual analysis. "
        "Strictly adhere to the output format and derive your answers ONLY from the data provided. "
        "Your final output MUST be a single, valid JSON object."
    )

    user_prompt = f"""
    Analyze the raw JSON data in the "Enriched_Call_Data" section. Perform these steps:
    1.  Iterate through each record. Extract numeric values from 'sessions_segments_media_streams_averagePacketLossRate', 'sessions_segments_media_streams_averageJitter', and 'sessions_segments_media_streams_averageRoundTripTime'.
    2.  Calculate the overall average for these three metrics across all records.
    3.  Identify unique devices from 'organizer_device' and 'participants_device'.
    4.  Based on your calculations, generate a structured JSON analysis with the following fields:
        - "call_summary": A one-sentence summary, mentioning users by display name.
        - "key_metrics": An object with the CALCULATED averages for packet loss, jitter, and RTT.
        - "insights": A list of detailed observations. Explicitly mention device models and observed metric values.
        - "actionable_insights": A list of specific, numbered steps for an administrator.
        - "sentiment": A one-word summary ("Poor", "Good", "Fair") based on metrics.

    Enriched_Call_Data:
    {json.dumps(call_records, indent=2)}
    """

    # 3. --- Call Azure OpenAI to Generate Insights with Profiling ---
    try:
        openai_client = AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version="2024-02-01"
        )
        
        # Start the timer just before the OpenAI call
        start_time_openai = time.perf_counter()
        
        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2000
        )
        
        # Stop the timer immediately after the call completes
        end_time_openai = time.perf_counter()
        openai_duration = end_time_openai - start_time_openai

        insights_json = json.loads(response.choices[0].message.content)

        # Add the profiling data to the final JSON output
        insights_json['profiling_metrics'] = {
            "azure_search_query_seconds": round(search_duration, 4),
            "openai_analysis_seconds": round(openai_duration, 4)
        }
        
        return insights_json

    except Exception as e:
        return {"error": f"Failed to get insights from Azure OpenAI: {e}"}

# --- Main execution block ---
if __name__ == "__main__":
    # --- HARDCODED VARIABLES FOR TESTING ---
    ORGANIZER_USER_ID = "a13e5e078-aec5-4345-a630-1ea859a1555a"
    PARTICIPANT_USER_ID = "41a95390-2c05-4837-85f4-542d4249f721"
    
    required_vars = [SEARCH_ENDPOINT, SEARCH_API_KEY, SEARCH_INDEX_NAME, OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_DEPLOYMENT_NAME]
    if not all(required_vars):
        print("Error: One or more environment variables are not set. Please check your .env file.")
    else:
        print(f"Analyzing call data for Organizer: {ORGANIZER_USER_ID} and Participant: {PARTICIPANT_USER_ID}...")
        print("-" * 50)

        analysis_result = get_call_analysis(ORGANIZER_USER_ID, PARTICIPANT_USER_ID)

        print(json.dumps(analysis_result, indent=2))
