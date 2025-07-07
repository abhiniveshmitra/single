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
    Queries the 'teams-calls' index with an enriched set of fields and uses an
    LLM to generate a detailed, structured JSON analysis.

    Args:
        organizer_user_id: The user ID of the call organizer.
        participant_user_id: The user ID of the call participant to analyze.

    Returns:
        A dictionary containing the AI-generated analysis or an error message.
    """
    # 1. --- Query Azure AI Search with Enriched Fields ---
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
            "startDateTime", "endDateTime"
        ]

        results = search_client.search(
            search_text="*",
            filter=filter_query,
            select=",".join(select_fields),
            include_total_count=True,
            top=10
        )

        call_records = [result for result in results]

        if not call_records:
            return {"error": "No call records found for the specified users."}

    except Exception as e:
        return {"error": f"Failed to query Azure AI Search: {e}"}

    # 2. --- Craft the Expert Prompt for the LLM ---
    system_prompt = (
        "You are an expert AI analyst specializing in Microsoft Teams call quality diagnostics. "
        "Your task is to analyze enriched call data from an Azure AI Search index. "
        "You must identify root causes for poor call quality by correlating user, device, and call time information. "
        "Your final output MUST be a single, valid JSON object and nothing else."
    )

    user_prompt = f"""
    Analyze the following call records for calls between organizer '{organizer_user_id}' (organizer) and '{participant_user_id}' (participant).
    The data includes user display names, device information, and call timestamps.
    Based on all available data, generate a structured JSON analysis. The JSON object must contain the following fields:
    - "call_summary": A brief, one-sentence overview of the findings, mentioning user display names.
    - "key_metrics": An object containing aggregated key metrics (average packet loss, jitter, etc., if available in the full data). If not available, state that.
    - "insights": A detailed list of observations. Correlate issues to specific devices or times if patterns exist.
    - "actionable_insights": A list of specific, numbered steps for an administrator. Mention devices or users by name.
    - "sentiment": A one-word summary of the user experience (e.g., "Poor", "Good", "Fair").

    Enriched Call Data:
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
            temperature=0.5,
            max_tokens=1500
        )

        insights_json = json.loads(response.choices[0].message.content)
        return insights_json

    except Exception as e:
        return {"error": f"Failed to get insights from Azure OpenAI: {e}"}

# --- Main execution block ---
if __name__ == "__main__":
    # --- HARDCODED VARIABLES FOR TESTING ---
    # These values are taken from the first record in your screenshot to ensure a match is found.
    # You can change these to analyze other users.
    ORGANIZER_USER_ID = "A13c6e9d-eb3b-43d5-a610-1ea859a15f6e"  # User: Morgan Patel
    PARTICIPANT_USER_ID = "41a95390-2c05-4837-85f4-542d4249f721"  # User: Bruce Messi
    
    # Check if all required environment variables are loaded from the .env file
    required_vars = [SEARCH_ENDPOINT, SEARCH_API_KEY, SEARCH_INDEX_NAME, OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_DEPLOYMENT_NAME]
    if not all(required_vars):
        print("Error: One or more environment variables are not set. Please check your .env file.")
    else:
        print(f"Analyzing call data for Organizer: {ORGANIZER_USER_ID} and Participant: {PARTICIPANT_USER_ID}...")
        print("-" * 50)

        # Call the function with the hardcoded variables
        analysis_result = get_call_analysis(ORGANIZER_USER_ID, PARTICIPANT_USER_ID)

        # Print the final JSON output with indentation
        print(json.dumps(analysis_result, indent=2))
