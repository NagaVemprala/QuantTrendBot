import requests
from dotenv import load_dotenv
import os

# Import required langchain components
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@tool("fetch_weather_data", description="Fetches weather data for a given location.", return_direct=False)
def fetch_weather_data(location: str) -> str:
    response = requests.get(f"https://wttr.in/{location}?format=j1")
    return response.json()

weather_agent = create_agent(
    model = "gpt-5-nano",
    tools = [fetch_weather_data],
    system_prompt = "You are a helpful assistant that can fetch weather data for any location using the provided tool."
)
response = weather_agent.invoke({
    "messages": [HumanMessage(content="What is the weather in New York?")]
})
print(response["messages"][-1].content)

