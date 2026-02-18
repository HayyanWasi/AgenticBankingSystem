from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv(override=True)


gemini_model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
def weather_agent(city: str)-> str:

      """Get weather for a given city."""

      return f"The weather in {city} is sunny with a high of 25°C and a low of 15°C."


agent = create_agent(
    model=gemini_model,
    tools=[weather_agent],
    system_prompt="you are a weather assistant agent, answer in 1 word"

)

response = agent.invoke(
      {

        "messages":[
              {
                "role":"user",
                "content": "what is the weather in new york?"}
        ]
    }
)

print(response["messages"][-1].content)






