import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from tools.kyc_tools import verify_id
from typing import Literal
load_dotenv(override=True)

supervisor_generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY2"),  # routing logic — KEY2
)
class RouteDecision(BaseModel):
    destination: Literal["loan_agent", "payment_tranaction_process_agent", "kyc_agent"]


route_decision_llm = supervisor_generator_llm.with_structured_output(RouteDecision)