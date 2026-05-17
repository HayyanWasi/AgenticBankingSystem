import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from app.tools.kyc_tools import verify_id

load_dotenv(override=True)

# # gemini-2.0-flash: 200 req/day free tier (vs 20/day for 2.5-flash)
# kyc_generator_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.7,
#     google_api_key=os.getenv("GOOGLE_API_KEY2"),  # agent processing — KEY2
# )

kyc_generator_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001", 
    temperature=0.7,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter
    }
)

kyc_conversation_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0.4,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)

kyc_extraction_llm_base = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)

tools = [verify_id]
llm_with_tools = kyc_generator_llm.bind_tools(tools)
tool_node = ToolNode(tools)
