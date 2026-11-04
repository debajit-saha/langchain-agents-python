
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Must be set BEFORE importing AzureSearch; it reads this env var at import time
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "contentVector"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import AzureSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Embeddings - must match the model used to create the index
# Use OpenAIEmbeddings (not AzureOpenAIEmbeddings) since Foundry exposes an OpenAI-compatible v1 endpoint
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    dimensions=3072
)

# Azure AI Search
vectorstore = AzureSearch(
    azure_search_endpoint="https://aisearch85.search.windows.net",
    azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    index_name="employee-handbook-index",
    embedding_function=embeddings.embed_query
)

retriever = vectorstore.as_retriever(
    search_type="similarity"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer using only this context.
If the answer is unavailable, say you don't know.

Context:
{context}

Question: {question}
""")

# Azure AI Foundry model
model = ChatOpenAI(
    base_url="https://foundry-dev-eus-902.services.ai.azure.com/openai/v1/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-4.1-mini"
)

# RAG Chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# Execute
print(rag_chain.invoke("How many annual leave days?")) 