import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

# 1. Load Data from Excel (Replaces the hardcoded list)
df = pd.read_excel("Airbnb_reviews.xlsx")
# We use 'comments' for the content. Change to 'description' or combine as needed.
text_input = df['comments'].dropna().astype(str).tolist()

# 2. Standard RAG Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(text_input, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Create Tool and Agent
# Note: Added a name 'airbnb_search' as agents require a non-empty tool name
retrieve_tool = create_retriever_tool(
    retriever, 
    name="airbnb_review_search", 
    description="Searches through Airbnb guest reviews and property descriptions."
)

agent = create_agent(
    model="gpt-5-nano",
    tools=[retrieve_tool],
    system_prompt="You are a helpful assistant that analyzes Airbnb reviews to help users make travel decisions."
)

# 4. Execute
response = agent.invoke({
    "messages": [HumanMessage(content="What are guests saying about the cleanliness and location of these listings?")]
})

print(response["messages"][-1].content)