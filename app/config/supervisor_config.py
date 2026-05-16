import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from app.tools.kyc_tools import verify_id
from typing import Literal
load_dotenv(override=True)

supervisor_generator_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0.7,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)
class RouteDecision(BaseModel):
    destination: Literal["loan_agent", "payment_tranaction_process_agent", "kyc_agent"]


route_decision_llm = supervisor_generator_llm.with_structured_output(RouteDecision)