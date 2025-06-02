import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Define schemas for all 4 arithmetic operations
functions = [
    {
        "name": "add",
        "description": "Add two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "subtract",
        "description": "Subtract two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "divide",
        "description": "Divide two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    },
]

# Define the basic arithmetic functions
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return "Division by zero" if b == 0 else a / b

print("Welcome to Arithmetic Chatbot! (Type 'exit' to quit)")

while True:
    user_input = input("\nUser: ")
    if user_input.strip().lower() == "exit":
        print("Bye!")
        break

    messages = [
        {"role": "system", "content": "You are a math assistant. If the user asks for arithmetic, use the available tools to get the result."},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto",  # Let the model decide
        max_tokens=200
    )

    choice = response.choices[0]

    # Check for a function call
    if hasattr(choice.message, "function_call") and choice.message.function_call:
        func_call = choice.message.function_call
        func_name = func_call.name
        args = json.loads(func_call.arguments)

        # Perform the correct arithmetic operation
        if func_name == "add":
            result = add(args["a"], args["b"])
        elif func_name == "subtract":
            result = subtract(args["a"], args["b"])
        elif func_name == "multiply":
            result = multiply(args["a"], args["b"])
        elif func_name == "divide":
            result = divide(args["a"], args["b"])
        else:
            result = "Unknown function."

        # Send function result back to the model for a friendly answer
        messages.append({
            "role": "function",
            "name": func_name,
            "content": str(result)
        })
        response2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200
        )
        print("\nAI Assistant:", response2.choices[0].message.content)
    else:
        print("\nAI Assistant:", choice.message.content)
