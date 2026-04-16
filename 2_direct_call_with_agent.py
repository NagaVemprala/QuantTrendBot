from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Setup Memory (Checkpointer) 
# This is what makes it an "Agent" that can remember previous turns
memory = InMemorySaver()

# 2. Create the Agent
# Even if you have no tools yet, create_agent sets up the Graph architecture
analytics_agent = create_agent(
    model="gpt-5.4", 
    tools=[],
    system_prompt="You are a helpful assistant. Please do not exceed more than 100 words in your response.",
    checkpointer=memory
)

# 3. Invoke the Agent
# Note the use of 'config' to track the conversation thread
config = {"configurable": {"thread_id": "analytics_session_1"}}

# we could do something like:
# --- THREAD 1: Analytics ---
config_analytics = {"configurable": {"thread_id": "analytics_thread"}}

# --- THREAD 2: Remote Work ---
config_remote = {"configurable": {"thread_id": "remote_work_thread"}}

print("--"*50)
# First Call
response = analytics_agent.invoke(
    {"messages": [HumanMessage(content="Can you explain about the linear programming?")]},
    config=config_analytics
)
# PRINTING HERE to verify the first run
print("First Response (Concept):", response["messages"][-1].content)

print("--"*50)
# Second Call
response = analytics_agent.invoke(
    {"messages": [HumanMessage(content="Why is it called programming eventhough it is not like a computer programming?")]},
    config=config_analytics
)
# This will now include the context of Linear Programming automatically
print("Second Response (Constraints):", response["messages"][-1].content)