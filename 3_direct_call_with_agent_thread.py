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
    model="gpt-5-nano", 
    tools=[],
    system_prompt="You are a helpful assistant that can help me with some Business Analytics tasks.",
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

# 4. Invokve the agent in loop based on the thread we want to use using a Input function,
# for example, we can have a simple input function that takes the thread name and the user query and invokes the agent with the appropriate config.
for _ in range(3):
    thread_name = input("Enter the thread name (analytics_thread or remote_work_thread): ")
    user_query = input("Enter your query: ")
    if thread_name == "analytics_thread":
        response = analytics_agent.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config_analytics
        )
    elif thread_name == "remote_work_thread":
        response = analytics_agent.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config_remote
        )
    else:
        print("Invalid thread name. Please try again.")
        continue

response = analytics_agent.invoke(
    {"messages": [HumanMessage(content="Can you explain about the linear programming?")]},
    config=config
)

print(response["messages"][-1].content)