from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

generator_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0.7,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)

class FruadCheck(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0, description="Fraud score between 0 and 1")
    is_fraud: bool

structured_evaluator_llm = generator_llm.with_structured_output(FruadCheck)