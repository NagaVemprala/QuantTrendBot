from dotenv import load_dotenv
import os

# Import required langchain components
from langchain.chat_models import init_chat_model

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = init_chat_model("gpt-5-nano", api_key=OPENAI_API_KEY)

response = model.invoke([
    {"role": "system", "content": "You are a helpful assistant that can help me with some Business Analytics tasks."},
    {"role": "user", "content": "Can you explain about the linear programming?"},
])  

response = model.invoke([
    {"role": "user", "content": "How are constraints created. Can you give me an example in simple terms?"},
])  

print(response.content) 
