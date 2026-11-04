
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class InterviewQuestion(BaseModel):
    technology: str = Field(description="Technology being discussed")
    difficulty: str = Field(description="Question difficulty")
    question: str = Field(description="Interview question")
    answer: str = Field(description="Explanation of the answer")


prompt = ChatPromptTemplate.from_template(
    "Generate one interview question about {technology} "
    "at {difficulty} difficulty. Include a clear answer."
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

model = ChatOpenAI(
    base_url=endpoint,
    api_key=api_key,
    model=deployment_name
)

# Ask the model for output matching the schema
structured_model = model.with_structured_output(InterviewQuestion)

# Build the chain
chain = prompt | structured_model

result = chain.invoke({
    "technology": "LangGraph",
    "difficulty": "intermediate"
})

print("Question:", result.question)
print("Answer:", result.answer)
print("Model dump:", result.model_dump())
