
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 1. Connect to your Azure AI Foundry deployment
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

model = ChatOpenAI(
    base_url=endpoint,
    api_key=api_key,
    model=deployment_name
)

# 2. Create a reusable prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms with one real-world example."
)

# 3. Connect the prompt to the model
chain = prompt | model

# 4. Execute the chain
response = chain.invoke({"topic": "RAG"})

print(response.content)
