import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

# --- Langchain & LangGraph Core Imports ---
from langchain_core.messages import BaseMessage, ToolMessage
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Azure Service Imports ---
from langchain_openai import AzureChatOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Load environment variables from .env file
load_dotenv()

# --- 1. Custom Tool Definition ---
def search_vdi_data_for_user(user_id: str) -> str:
    """
    Searches the 'teams-calls' Azure AI Search index for a specific user's
    call records and returns data relevant to VDI performance analysis.
    The user_id can be a user principal name (UPN) or similar identifier.
    """
    try:
        print(f"---TOOL: Searching for user: {user_id}---")
        search_client = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ["AZURE_SEARCH_INDEX"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
        )

        select_fields = [
            "startDateTime", "endDateTime", "callOverallStatus",
            "audioQuality_Score", "videoQuality_Score",
            "organizer_user", "sessions_caller_identity_device", "sessions_callee_identity_device",
            "sessions_segments_media_streams_averageRoundTripTime",
            "sessions_segments_media_streams_averageVideoPacketLossRate",
            "sessions_segments_media_streams_averageAudioNetworkJitter",
            "sessions_segments_media_streams_cpuInsufficentEventRatio"
        ]

        results = search_client.search(
            search_text="*",
            filter=f"organizer_user eq '{user_id}' or sessions_caller_identity_device eq '{user_id}' or sessions_callee_identity_device eq '{user_id}'",
            select=",".join(select_fields),
            top=10,
            order_by="startDateTime desc"
        )

        call_records = [result for result in results]

        if not call_records:
            return f"No call records found for user '{user_id}'."

        print(f"---TOOL: Found {len(call_records)} records for {user_id}---")
        return json.dumps(call_records)

    except Exception as e:
        return f"An error occurred while searching: {str(e)}"

# --- 2. Graph State Definition ---
class VdiAnalysisState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 3. Agent and Graph Construction ---
AGENT_SYSTEM_PROMPT = """
You are an expert performance analyst specializing in Virtual Desktop Infrastructure (VDI)
and Microsoft Teams. Your goal is to analyze Teams Call Detail Records (CDR) to identify
root causes for poor user experiences and provide actionable insights for IT administrators.

When you receive a prompt about a user, you must use the 'search_vdi_data_for_user'
tool to retrieve their recent call data from the Azure AI Search index.

After retrieving the data, analyze it carefully. Pay close attention to these key VDI performance indicators:
- `cpuInsufficentEventRatio`: A high value (e.g., > 0.1) is a strong indicator that the user's VDI session is under-resourced (CPU bottleneck).
- `averageRoundTripTime`: High latency (e.g., > 100ms) can be a network issue or VDI display protocol lag.
- `averageVideoPacketLossRate` and `averageAudioNetworkJitter`: These point to network instability.

Based on your analysis, formulate a final answer with two sections:
1.  **Analysis Summary:** A brief, clear summary of the findings from the data.
2.  **Actionable Insights:** A list of concrete, numbered steps that an IT administrator should take to investigate and resolve the issue.
"""

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
    streaming=True
)

tools = [search_vdi_data_for_user]
tool_node = ToolNode(tools)

# *** THIS SECTION IS CORRECTED ***
# The prompt now includes the required 'agent_scratchpad' placeholder.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt)

def agent_node(state: VdiAnalysisState):
    result = agent.invoke(state)
    return {"messages": result}

workflow = StateGraph(VdiAnalysisState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

def should_continue(state: VdiAnalysisState):
    if isinstance(state['messages'][-1], ToolMessage):
        return "agent"
    return END

workflow.add_conditional_edges(
    "agent",
    lambda x: "tools" if x['messages'][-1].tool_calls else END,
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- 4. Running the Agent ---
if __name__ == "__main__":
    print("VDI Analysis Agent is ready. Enter a username to analyze.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("User> ")
        if user_input.lower() == "exit":
            break
        
        inputs = {"messages": [("user", user_input)]}
        print("\nAI> ", end="", flush=True)
        # The agent executor logic from LangGraph handles the agent_scratchpad internally.
        # We only need to stream the final response content.
        for chunk in app.stream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            if message.content:
                # Only print content from the AI's final response, not tool calls.
                if not isinstance(message, ToolMessage) and not message.tool_calls:
                     print(message.content, end="", flush=True)
        print("\n")

