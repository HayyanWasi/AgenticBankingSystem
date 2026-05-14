import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from tools.kyc_tools import verify_id

load_dotenv(override=True)

# gemini-2.0-flash: 200 req/day free tier (vs 20/day for 2.5-flash)
kyc_generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY2"),  # agent processing — KEY2
)

kyc_conversation_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY1"),  # user-facing conversation — KEY1
)

kyc_extraction_llm_base = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY2"),  # structured extraction — KEY2
)

tools = [verify_id]
llm_with_tools = kyc_generator_llm.bind_tools(tools)
tool_node = ToolNode(tools)
