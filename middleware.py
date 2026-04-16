"""Middleware for handling RAG application logic.
In LangChain 1.0 and LangGraph, Middleware (often implemented via "Wrappers" or "Hooks") serves as an interceptor layer 
that allows you to execute logic before or after a call to a Chat Model, Tool, or Agent.
It sits between the user input and the model's response, enabling you to manipulate inputs, outputs, or even the behavior of the model itself.
In the context of a RAG application, Middleware can be used to:
1. Pre-process user queries to extract relevant information or reformat them for better retrieval.
2. Post-process the retrieved documents to filter, summarize, or re-rank them before they are fed into the language model.
3. Implement custom logic for handling edge cases, such as when no relevant documents are found or when the retrieved information is too verbose.
4. Log interactions for monitoring and debugging purposes.
5. Integrate additional tools or APIs that can enhance the retrieval or generation process, such as external databases, knowledge graphs, or specialized APIs for specific domains.
"""


from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from langchain.messages import HumanMessage
from dataclasses import dataclass

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def custom_system_prompt(context: Context) -> str:
    if context.user_role == "data_scientist":
        return "You are a helpful assistant that provides detailed technical insights and code examples."
    elif context.user_role == "business_analyst":
        return "You are a helpful assistant that provides high-level summaries and actionable business insights."
    else:
        return "You are a helpful assistant that provides general information and guidance."
    
agent = create_agent(
    model = "gpt-5-nano",
    tools = [],
    context_schema=Context
)

# Example of invoking the agent with a specific user role
response = agent.invoke({
    "messages": [HumanMessage(content="What are some key insights on remote work and cost optimization?")],
    "context": Context(user_role="data_scientist")
})
print(response["messages"][-1].content)
