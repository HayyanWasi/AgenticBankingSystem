from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

class FruadCheck(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0, description="Fraud score between 0 and 1")
    is_fraud: bool

structured_evaluator_llm = generator_llm.with_structured_output(FruadCheck)
