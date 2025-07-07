import os
import json
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Configuration for Azure Services ---
# The script will now load these values from your .env file.
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def get_call_analysis(organizer_user_id: str, participant_user_ids: list[str]) -> dict:
    """
    Queries the 'teams-calls' index for calls involving multiple participants and
    uses an LLM to generate actionable insights in a structured JSON format.

    Args:
        organizer_user_id: The user ID of the call organizer.
        participant_user_ids: A list of user IDs of participants to analyze.

    Returns:
        A dictionary containing the AI-generated analysis or an error message.
    """
    # 1. --- Query Azure AI Search ---
    try:
        search_credential = AzureKeyCredential(SEARCH_API_KEY)
        search_client = SearchClient(endpoint=SEARCH_ENDPOINT,
                                     index_name=SEARCH_INDEX_NAME,
                                     credential=search_credential)

        # Construct the filter for multiple participants using the 'any' and 'search.in' functions.
        # This assumes 'participants_user_id' is a Collection(Edm.String) in your search index.
        participants_filter_str = ",".join(f"'{p}'" for p in participant_user_ids)
        filter_query = (
            f"organizer_user_id eq '{organizer_user_id}' and "
            f"participants_user_id/any(p: search.in(p, {participants_filter_str}, ','))"
        )

        results = search_client.search(
            search_text="*",
            filter=filter_query,
            include_total_count=True,
            top=10  # Retrieve top 10 relevant call records for context
        )

        call_records = [result for result in results]

        if not call_records:
            return {"error": "No call records found for the specified users."}

    except Exception as e:
        return {"error": f"Failed to query Azure AI Search: {e}"}

    # 2. --- Craft the Expert Prompt for the LLM ---
    system_prompt = (
        "You are an expert AI analyst specializing in Microsoft Teams call quality and network diagnostics. "
        "Your task is to analyze raw call data from an Azure AI Search index named 'teams-calls'. "
        "You must identify root causes for poor call quality and provide clear, actionable recommendations. "
        "Your final output MUST be a single, valid JSON object and nothing else."
    )

    user_prompt = f"""
    Analyze the following call records for calls organized by '{organizer_user_id}' involving any of the participants: {', '.join(participant_user_ids)}.
    Based on the data, generate a structured JSON analysis. The JSON object must contain the following fields:
    - "call_summary": A brief, one-sentence overview of the findings.
    - "key_metrics": An object containing aggregated key metrics like average packet loss, jitter, and round-trip time.
    - "insights": A detailed list of observations and identified issues (e.g., "High packet loss observed across multiple sessions, indicating potential network congestion.").
    - "actionable_insights": A list of specific, numbered steps for a network administrator to take (e.g., "1. Investigate the network segment for user X.", "2. Check for firmware updates on headset model Y.").
    - "sentiment": A one-word summary of the user experience (e.g., "Poor", "Good", "Fair").

    Call Data:
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
    # Check if all required environment variables are loaded
    required_vars = [SEARCH_ENDPOINT, SEARCH_API_KEY, SEARCH_INDEX_NAME, OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_DEPLOYMENT_NAME]
    if not all(required_vars):
        print("Error: One or more environment variables are not set. Please check your .env file and ensure it is in the same directory.")
    else:
        # Example Usage: Analyze calls organized by 'adele.vance@contoso.com'
        # involving 'alex.wilber@contoso.com' or 'megan.bowen@contoso.com'.
        organizer = "adele.vance@contoso.com"
        participants_to_check = ["alex.wilber@contoso.com", "megan.bowen@contoso.com"]

        print(f"Analyzing call data for Organizer: {organizer} and Participants: {', '.join(participants_to_check)}...")
        print("-" * 50)

        analysis_result = get_call_analysis(organizer, participants_to_check)

        # Print the final JSON output with indentation
        print(json.dumps(analysis_result, indent=2))


pip install python-dotenv azure-search-documents openai

# Azure AI Search configuration
AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net"
AZURE_SEARCH_API_KEY="your_search_api_key"
AZURE_SEARCH_INDEX_NAME="teams-calls"

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT="https://your-openai-service.openai.azure.com/"
AZURE_OPENAI_KEY="your_openai_api_key"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"


