# pip install -qU langchain "langchain[anthropic]"
import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key="AIzaSyBq9ILzSTELnw6SQRjxzS64ABp17lwyTUA"
    
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result=agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(result)