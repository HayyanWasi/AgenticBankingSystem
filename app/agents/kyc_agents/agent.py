from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
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
from langchain_ollama import ChatOllama
from langgraph.types import Command





load_dotenv(override=True)

local_llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
    # This ensures the model doesn't stay in RAM after the task is done
    keep_alive="0s" 
)


kyc_generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

kyc_agent_result = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY2"),
)


class UserContext(BaseModel):
    id_num: str = Field(description="Customer ID number")
    full_name: str = Field(description="Customer Full Name")


@tool("verify_customer_id", 
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
        # First call: create initial messages and include them in returned state
        initial_messages = [
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
        response = llm_with_tools.invoke(initial_messages)
        return {"messages": initial_messages + [response]}
    else:
        # Subsequent calls: full history already in state
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}


# structured_kyc_llm = kyc_agent_result.with_structured_output(VerificationResult)
# def kyc_agent_results(state: KYCState):

#     last_message = state["messages"][-1]

#     if isinstance(last_message.content, list):
#         content = last_message.content[0]['text']
#     else:
#         content = last_message.content

#     response = structured_kyc_llm.invoke([
#         SystemMessage(content="Extract the KYC verification result from this message."),
#         HumanMessage(content=content)
#     ])

#     return {"kyc_score": response.kyc_score,
#              "verification_status": response.status,
#                "rejection_reason": response.rejection_reason,
#             }



#To reduce LLM calls costs
def kyc_agent_results(state: KYCState):
    import json
    
    # find tool message in state
    for message in reversed(state['messages']):
        if hasattr(message, 'name') and message.name == 'verify_customer_id':
            tool_result = json.loads(message.content)
            score = tool_result['score']
            
            if score >= 0.8:
                status = 'approved'
            elif score >= 0.4:
                status = 'human_review'
            else:
                status = 'rejected'
            
            rejection_reason = ""
            if 'reason' in tool_result:
                rejection_reason = tool_result['reason']
            elif not tool_result.get('name_match'):
                rejection_reason = "Name does not match ID records."
            elif not tool_result.get('not_expired'):
                rejection_reason = "ID card is expired."
                
            return {
                "kyc_score": score,
                "verification_status": status,
                "rejection_reason": rejection_reason
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



def score_condition(state: KYCState):
    if state["kyc_score"] >= 0.8:
        return 'approve'
    elif state['kyc_score'] >= 0.4: 
         return 'human_review'
    else:
        return 'reject'


def  notify_customer(state: KYCState) -> dict:
    if state['kyc_score'] >= 0.8:
        message = f"Dear {state['full_name']}, your KYC is approved."
    elif state['human_decision'] == 'approve':
        message = f"Dear {state['full_name']}, your KYC is approved after manual review."
    else:
        message = f"Dear {state['full_name']}, your KYC is rejected. Reason: {state['rejection_reason']}"
    return {"notification_message": message, 
            "kyc_status": "completed"
            }


def human_review(state: KYCState) -> dict:
    decision = interrupt({
        'full_name': state['full_name'],
        'id_card_num': state['id_card_num'],
        'kyc_score': state['kyc_score'],
        'verification_status': state['verification_status'],    
        'question': 'Approve or reject this KYC?'
    })
    return {'human_decision': decision}

graph = StateGraph(KYCState)

graph.add_node("kyc_agent", kyc_agent)
graph.add_node("action", tool_node)
graph.add_node("kyc_agent_results", kyc_agent_results)
graph.add_node('human_review', human_review)
graph.add_node('notify_customer', notify_customer)


graph.add_edge(START, "kyc_agent")
graph.add_conditional_edges(
    "kyc_agent",
    tools_condition,
    {
        "action": "action",
        "results": "kyc_agent_results"
    }
)
graph.add_edge("action", "kyc_agent")
graph.add_conditional_edges(
    "kyc_agent_results",
    score_condition,
    {
        "approve": "notify_customer",
        "human_review": "human_review",
        "reject": "notify_customer"
    }
)
graph.add_edge("human_review", "notify_customer")
graph.add_edge("notify_customer", END)


checkPointer = MemorySaver()

workflow = graph.compile(checkpointer=checkPointer)

#TEST CASE 1: valid inputs

# initial_state = {
#     "id_card_num": "42101-1234567-1",
#     "full_name": "John Doe",
#     "phone_number": "0300-1234567",
#     "issue_date": "2020-01-01",
#     "expiry_date": "2028-01-01",
#     "nationality": "Pakistani",
#     "date_of_birth": "1990-01-01",
#     "place_of_birth": "Karachi",
#     "gender": "Male",
#     "kyc_score": 0.0,
#     "verification_status": "",
#     "human_decision": "",
#     "rejection_reason": "",
#     "notification_message": "",
#     "kyc_status": "",
#     "messages": []
# }

# config = {"configurable": {"thread_id": "kyc-test-1"}}
# result = workflow.invoke(initial_state, config=config)
# print(result)




#TEST CASE 2: human review 

# initial_state = {
#     "id_card_num": "42201-5544332-7",  # Fatima Ali
#     "full_name": "Fatima Ali",
#     "phone_number": "0300-1234567",
#     "issue_date": "2020-01-01",
#     "expiry_date": "2028-01-01",
#     "nationality": "Pakistani",
#     "date_of_birth": "1990-01-01",
#     "place_of_birth": "Karachi",
#     "gender": "Male",
#     "kyc_score": 0.0,
#     "verification_status": "",
#     "human_decision": "",
#     "rejection_reason": "",
#     "notification_message": "",
#     "kyc_status": "",
#     "messages": []
# }
# config2 = {"configurable": {"thread_id": "kyc-test-2b"}}
# result = workflow.invoke(initial_state, config=config2)
# result = workflow.invoke(Command(resume="reject"), config=config2)
# print("Score:", result['kyc_score'])
# print("Status:", result['verification_status'])
# print("Notification:", result['notification_message'])
# print("Human Decision:", result['human_decision'])
# print("REJECTED:", result['notification_message'])



#TEST CASE 3: 
initial_state = {
   "id_card_num": "00000-0000000-0",
    "full_name": "Nobody",
    "phone_number": "0300-1234567",
    "issue_date": "2020-01-01",
    "expiry_date": "2028-01-01",
    "nationality": "Pakistani",
    "date_of_birth": "1990-01-01",
    "place_of_birth": "Karachi",
    "gender": "Male",
    "kyc_score": 0.0,
    "verification_status": "",
    "human_decision": "",
    "rejection_reason": "",
    "notification_message": "",
    "kyc_status": "",
    "messages": []
}
config2 = {"configurable": {"thread_id": "kyc-test-2b"}}
result = workflow.invoke(initial_state, config=config2)
result = workflow.invoke(Command(resume="reject"), config=config2)
print("Score:", result['kyc_score'])
print("Status:", result['verification_status'])
print("Notification:", result['notification_message'])
print("Human Decision:", result['human_decision'])
print("REJECTED:", result['notification_message'])