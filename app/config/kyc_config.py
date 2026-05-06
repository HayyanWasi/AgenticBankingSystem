import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from app.tools.kyc_tools import verify_id

load_dotenv(override=True)

kyc_generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY1"),
)

tools = [verify_id]
tool_node = ToolNode(tools)
llm_with_tools = kyc_generator_llm.bind_tools(tools)
