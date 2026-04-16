from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
#from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import create_retriever_tool
from langchain.messages import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

text_input = [
    "The transition to a permanent remote work structure optimizes operational expenditure and empowers employees with greater scheduling autonomy.",
    "Decentralized work environments enhance talent acquisition by removing geographical barriers to high-skilled labor markets.",
    "Traditional office settings remain essential for maintaining organizational culture and preventing the erosion of social capital among junior associates.",
    "The absence of face-to-face interaction in virtual teams often results in communication silos and a measurable degradation of creative brainstorming.",
    "Implementing a Value-at-Risk (VaR) framework allows financial institutions to quantify the potential loss of a portfolio over a specific time horizon.",
    "Stochastic modeling techniques, such as the Black-Scholes equation, are fundamental for the accurate pricing of European-style options.",
    "Monte Carlo simulations provide a robust method for assessing the impact of demand volatility on long-term capital investment decisions.",
    "Just-in-time inventory management reduces holding costs but increases the vulnerability of the supply chain to sudden logistical disruptions.",
    "Predictive maintenance algorithms utilize sensor data to anticipate equipment failure, thereby minimizing unplanned downtime in manufacturing facilities.",
    "Global supply chain resilience is bolstered by diversifying tier-one suppliers across multiple geopolitical regions to mitigate localized shocks."
]

vectorstore = FAISS.from_texts(text_input, embeddings)

print("test the similarity search: Using 'Evaluating the remote work structure for cost optimization is important.'")
query = "Evaluating the remote work structure for cost optimization is important."
similar_docs = vectorstore.similarity_search(query, k=5)
for idx, doc in enumerate(similar_docs):
    print(f"Similar Document {idx+1}: {doc}")

print("--"*50)

query = "Investing in S&P 500 index options requires understanding of stochastic modeling techniques."
similar_docs = vectorstore.similarity_search(query, k=5)
for idx, doc in enumerate(similar_docs):
    print(f"Similar Document {idx+1}: {doc}")

retriever = vectorstore.as_retriever(kwargs={"k": 3})


retrieve_similar_texts = create_retriever_tool(retriever, 
                                               name = "retrieve_business_insights", 
                                               description="Retrieves texts similar to the query from the vector store.")

agent = create_agent(
    model = "gpt-5-nano",
    tools = [retrieve_similar_texts],
    system_prompt = "You are a helpful assistant that retrieves similar texts from a vector store based on the user's query."
)
response = agent.invoke({
    "messages": [HumanMessage(content="What are some key insights on remote work and cost optimization?")]
})
print(response["messages"][-1].content)




