import os
import json
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
    Queries the 'teams-calls' index with a comprehensive set of fields and uses an
    advanced LLM prompt to generate a detailed, data-driven JSON analysis.

    Args:
        organizer_user_id: The user ID of the call organizer.
        participant_user_id: The user ID of the call participant to analyze.

    Returns:
        A dictionary containing the AI-generated analysis or an error message.
    """
    # 1. --- Query Azure AI Search with a Comprehensive Field List ---
    try:
        search_credential = AzureKeyCredential(SEARCH_API_KEY)
        search_client = SearchClient(endpoint=SEARCH_ENDPOINT,
                                     index_name=SEARCH_INDEX_NAME,
                                     credential=search_credential)

        filter_query = (
            f"organizer_user_id eq '{organizer_user_id}' and "
            f"participants_user_id eq '{participant_user_id}'"
        )
        
        # IMPROVED: This list is now much more comprehensive. It fetches user info,
        # devices, timestamps, and likely call quality metric fields.
        # NOTE: You may need to adjust the metric field names (e.g., 'averagePacketLoss')
        # to match your exact index schema. Use Search explorer to verify them.
        select_fields = [
            # User and Device Info
            "organizer_user_id", "organizer_user_displayName", "organizer_device",
            "participants_user_id", "participants_user_displayName", "participants_device",
            # Timestamps
            "startDateTime", "endDateTime",
            # Key Call Quality Metrics (adjust names if needed)
            "averagePacketLoss", "averageJitter", "averageRoundTripTime", "averageLatency",
            "maxPacketLoss", "maxJitter", "maxRoundTripTime", "maxLatency"
        ]

        results = search_client.search(
            search_text="*",
            filter=filter_query,
            select=",".join(select_fields),
            include_total_count=True,
            top=50  # Get more records for a better analysis
        )

        call_records = [result for result in results]

        if not call_records:
            return {"error": "No call records found for the specified users."}

    except Exception as e:
        return {"error": f"Failed to query Azure AI Search: {e}"}

    # 2. --- Craft a More Intelligent and Directive Prompt ---
    system_prompt = (
        "You are a meticulous AI data analyst for Microsoft Teams call quality. Your task is to process a JSON array of call records. "
        "You must perform calculations on the provided data and present a factual analysis. "
        "Strictly adhere to the output format and derive your answers ONLY from the data provided. "
        "Your final output MUST be a single, valid JSON object."
    )

    user_prompt = f"""
    Analyze the raw JSON data provided in the "Enriched_Call_Data" section below. Perform the following steps:
    1.  Iterate through each call record in the JSON array.
    2.  From each record, extract the numeric values for 'averagePacketLoss', 'averageJitter', and 'averageRoundTripTime'.
    3.  Calculate the overall average for these metrics across all records. If data is missing for a metric, state "Not available".
    4.  Identify all unique device models listed in the 'organizer_device' and 'participants_device' fields.
    5.  Based on your calculations and data extraction, generate a structured JSON analysis. The JSON object must contain the following fields:
        - "call_summary": A one-sentence overview of the findings, including user display names.
        - "key_metrics": An object containing the CALCULATED overall averages for packet loss, jitter, and RTT.
        - "insights": A list of detailed observations. Explicitly mention the device models you found. If quality metrics were high, point it out.
        - "actionable_insights": A list of specific, numbered steps for an administrator, referencing the specific data found.
        - "sentiment": A one-word summary of the user experience based on the calculated metrics (e.g., "Poor", "Good", "Fair").

    Enriched_Call_Data:
    {json.dumps(call_records, indent=2)}
    """

    # 3. --- Call Azure OpenAI to Generate Insights ---
    try:
        openai_client = AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version="2024-02-01"
        )
        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,  # Lower temperature for more factual, less creative responses
            max_tokens=2000
        )

        insights_json = json.loads(response.choices[0].message.content)
        return insights_json

    except Exception as e:
        return {"error": f"Failed to get insights from Azure OpenAI: {e}"}

# --- Main execution block ---
if __name__ == "__main__":
    # --- HARDCODED VARIABLES FOR TESTING ---
    # These values are from the successful run in your screenshot.
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
