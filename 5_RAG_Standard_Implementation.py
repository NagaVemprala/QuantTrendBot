import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Setup Embeddings and Vectorstore
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Define the Prompt Template
# LangChain 1.0 encourages using explicit templates for RAG
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Initialize the Model
model = init_chat_model("gpt-5-mini", model_provider="openai", api_key=OPEN_AI_API_KEY)

# 4. Build the RAG Chain using LCEL
# This replaces the Agent loop with a fixed, linear pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 5. Execution
query = "What are some key insights on remote work and cost optimization?"
response = rag_chain.invoke(query)

print(f"Question: {query}")
print(f"Answer: {response}")