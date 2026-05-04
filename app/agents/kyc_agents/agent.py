from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from datetime import datetime
import os
from dotenv import load_dotenv

from app.agents.kyc_agents.mockdata import MOCK_DB
from app.schemas.kyc_agent_schema import KYCState, VerificationResult

from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
kyc_generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


class UserContext(BaseModel):
    id_num: str = Field(required=True, description="Customer ID number")
    full_name: str = Field(required=True, description="Customer Full Name")


@tool("Verify customer ID from database",
 description="verify the customer id from the database. if not found, dismiss verification",
 args_schema=UserContext)
def verify_id(id_num: str, full_name: str) -> dict:
    """Search customer in DB, verify ID and check expiry in one call"""
    customer = MOCK_DB.get(id_num)
    
    if not customer:
        return {"score": 0.0, "reason": "Customer not found"}
    
    name_match = customer["full_name"].lower() == full_name.lower()
    # Fixed: use datetime.strptime since we imported the class
    not_expired = datetime.strptime(customer["expiry_date"], "%Y-%m-%d") > datetime.now()
    
    score = 0.0
    if name_match: score += 0.4
    if not_expired: score += 0.4
    score += 0.2  # found in DB
    
    return {"score": score, "name_match": name_match, "not_expired": not_expired}



tools = [verify_id]

tool_node = ToolNode(tools)
llm_with_tools = kyc_generator_llm.bind_tools(tools)


def kyc_agent(state: KYCState):
    """
    get the tool result and call the llm for final results
    """

    messages = state['messages']
    if not messages: 
        messages= [
        SystemMessage(
            content=f"""
            ##You are an Know Your customer(KYC) verification agent.
            Verify this customer:
            ID: {state['id_card_num']}
            Full Name: {state['full_name']}


            Use your tools to get the final score and verification result.
            if the score is less than 0.4 reject the customer
            if the score is between 0.4 and 0.7, ask for human review
            if the score is greater than 0.7 approve the customer
            """
        ),
        HumanMessage(
            content="""
            verify this customer using your tools. 
            """
        )
    ]



    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


structured_kyc_llm = kyc_generator_llm.with_structured_output(VerificationResult)


def kyc_agent_results(state: KYCState):

    last_message = state["messages"][-1]

    response = structured_kyc_llm.invoke([
        SystemMessage(content="Extract the KYC verification result from this message."),
        HumanMessage(content=str(last_message.content))
    ])

    return {"kyc_score": response.kyc_score,
             "verification_status": response.status,
               "rejection_reason": response.rejection_reason,
            }


def tools_condition(state: KYCState):
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM wants to use a tool, go to the "action" node
    #Human ne query di example user ki id nikal k lao -> AI ne bola i dont know the answer i have specific tool 
    #to verify it Ai calls tool using tool_calls is s technically an attribute (a piece of data) of the AIMessage object
    # action is attached with this node graph.add_node("action", tool_node) graph is calling the tool_node


    if last_message.tool_calls:
        return "action"
    #if we dont need tool we just simply hit the results END node 
    # Otherwise, we are done with tools, go to extract final results
    return "results" 


graph = StateGraph(KYCState)

graph.add_node("kyc_agent", kyc_agent)
graph.add_node("action", tool_node)
graph.add_node("kyc_agent_results", kyc_agent_results)

graph.add_edge(START, "kyc_agent")
graph.add_conditional_edges(
    "kyc_agent",
    tools_condition,
    {
        "tools": "action",
        "results": "kyc_agent_results"
    }
)
graph.add_edge("action", "kyc_agent")
graph.add_edge("kyc_agent_results", END)
