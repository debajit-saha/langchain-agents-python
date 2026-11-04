# pip install langchain langchain-openai python-dotenv
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

model = ChatOpenAI(
    base_url=endpoint,
    api_key=api_key,
    model=deployment_name
)

response = model.invoke(
    "What are the three main benefits of using managed AI endpoints in the cloud?")
print(response.content)
