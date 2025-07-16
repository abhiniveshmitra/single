import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, ToolMessage
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Tool for Azure AI Search ---
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Load environment variables from .env file
load_dotenv()

# --- 1. Custom Tool Definition ---
# This tool queries the 'teams-calls' index for a specific user.

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

        # Fields to retrieve, based on provided schema images[2][3][4]
        # Focus on identity, quality scores, and performance metrics
        select_fields = [
            "startDateTime", "endDateTime", "callOverallStatus",
            "audioQuality_Score", "videoQuality_Score",
            "organizer_user", "sessions_caller_identity_device", "sessions_callee_identity_device",
            "sessions_segments_media_streams_averageRoundTripTime",
            "sessions_segments_media_streams_averageVideoPacketLossRate",
            "sessions_segments_media_streams_averageAudioNetworkJitter",
            "sessions_segments_media_streams_cpuInsufficentEventRatio" # Critical for VDI
        ]

        # Search for the user as either organizer, caller, or callee
        results = search_client.search(
            search_text="*", # Match all documents
            filter=f"organizer_user eq '{user_id}' or sessions_caller_identity_device eq '{user_id}' or sessions_callee_identity_device eq '{user_id}'",
            select=",".join(select_fields),
            top=10 # Get the last 10 calls for analysis
        )

        call_records = [result for result in results]

        if not call_records:
            return f"No call records found for user '{user_id}'."

        # Return the results as a JSON string for the LLM to process
        return json.dumps(call_records)

    except Exception as e:
        return f"An error occurred while searching: {str(e)}"

# --- 2. Graph State Definition ---
# This class defines the structure that holds data as it moves through the agent.

class VdiAnalysisState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- 3. Agent and Graph Construction ---

# System prompt to define the agent's expertise and behavior
AGENT_SYSTEM_PROMPT = """
You are an expert performance analyst specializing in Virtual Desktop Infrastructure (VDI)
and Microsoft Teams. Your goal is to analyze Teams Call Detail Records (CDR) to identify
root causes for poor user experiences and provide actionable insights for IT administrators.

When you receive a prompt about a user, you must use the 'search_vdi_data_for_user'
tool to retrieve their recent call data from the Azure AI Search index.

After retrieving the data, analyze it carefully. Pay close attention to these key VDI performance indicators:
- `cpuInsufficentEventRatio`: A high value (e.g., > 0.1) is a strong indicator that the user's VDI session is under-resourced (CPU bottleneck).
- `averageRoundTripTime`: High latency can be a network issue or VDI display protocol lag.
- `averageVideoPacketLossRate` and `averageAudioNetworkJitter`: These point to network instability, which could be related to the VDI's network configuration or the user's connection.

Based on your analysis, formulate a final answer with two sections:
1.  **Analysis Summary:** A brief, clear summary of the findings from the data.
2.  **Actionable Insights:** A list of concrete, numbered steps that an IT administrator should take to investigate and resolve the issue.
"""

# Initialize the LLM from Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
    streaming=True
)

# Define the tools the agent can use
tools = [search_vdi_data_for_user]
tool_node = ToolNode(tools)

# Create the agent prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the agent itself by binding the tools to the LLM
agent = create_openai_tools_agent(llm, tools, prompt)

# Define the agent and tool nodes for the graph
def agent_node(state: VdiAnalysisState):
    """Invokes the agent to decide on the next action."""
    result = agent.invoke(state)
    return {"messages": result}

# Define the graph logic
workflow = StateGraph(VdiAnalysisState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

# Conditional edge: after the agent runs, check if it decided to call a tool
def should_continue(state: VdiAnalysisState):
    if isinstance(state['messages'][-1], ToolMessage):
        return "agent" # If a tool was just called, loop back to the agent for analysis
    # If there are no tool calls, the agent has finished
    return END

workflow.add_conditional_edges(
    "agent",
    lambda x: "tools" if x['messages'][-1].tool_calls else END,
)
workflow.add_edge("tools", "agent")


# Compile the graph into a runnable application
app = workflow.compile()

# --- 4. Running the Agent ---
if __name__ == "__main__":
    print("VDI Analysis Agent is ready. Enter a username to analyze.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("User> ")
        if user_input.lower() == "exit":
            break
        
        # Stream the agent's response
        inputs = {"messages": [("user", user_input)]}
        for event in app.stream(inputs, stream_mode="values"):
            message = event["messages"][-1]
            if isinstance(message, tuple): # Is a tool call/response
                pass # Don't print tool I/O to the user
            else:
                # Print the AI's final response chunk by chunk
                message.pretty_print()
