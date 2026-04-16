from dataclasses import dataclass

import requests
from dotenv import load_dotenv
import os

# Import required langchain components
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class user_context:
    user_id: str

@dataclass
class OutputFormat: 
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float

@tool('get_weather_data', description='Fetches weather data for a given location.', return_direct=False)
def get_weather_data(location: str) -> str:
    response = requests.get(f"https://wttr.in/{location}?format=j1")
    return response.json()

@tool('fetch_user_context', description='Fetches user context for a given user ID to fetch the location as string.')
def fetch_user_context(runtime: ToolRuntime[user_context]) -> str:
    # In a real application, this would fetch data from a database or another service.
    # Here, we will return a hardcoded location for demonstration purposes.
    user_id = runtime.context.user_id
    user_context_data = {
        "user_123": "New York",
        "user_456": "Los Angeles",
        "user_789": "Chicago"
    }
    return user_context_data.get(user_id, "Unknown Location")

checkpointerMemory = InMemorySaver()

weather_agent = create_agent(
    model = "gpt-5-nano",
    tools = [get_weather_data, fetch_user_context],
    system_prompt = "You are a helpful assistant that can fetch weather data. Use the tool 'fetch_user_context' to get the user's location based on their user ID, and then use the 'get_weather_data' tool to fetch the weather data for that location.",
    checkpointer = checkpointerMemory,
    context_schema=user_context,
    response_format=OutputFormat
)

config_settings = {'configurable': {'thread_id': 0}}

response = weather_agent.invoke({
    "messages": [
    {
        "role": "user",
        "content": "What is the weather for my current location?"
    }
],
    },
    context = user_context(user_id="user_123"),
    config = config_settings
)
print(response["messages"][-1].content)