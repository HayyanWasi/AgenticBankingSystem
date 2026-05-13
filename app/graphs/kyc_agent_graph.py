import os
from app.agents.kyc_agents.notify_customer import notify_customer
from app.agents.kyc_agents.human_review_kyc import human_review
from app.agents.kyc_agents.kyc_agent import kyc_agent
from app.agents.kyc_agents.kyc_agent_results import kyc_agent_results
from app.schemas.kyc_agent_schema import KYCState
from app.conditions.kyc_process_conditions.kyc_tool_conditions import tools_condition
from app.conditions.kyc_process_conditions.score_conditions import score_condition
from app.agents.kyc_agents.upsert_kycrecord import upsert_kyc_record, route_after_upsert
from app.config.kyc_config import tool_node
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

load_dotenv(override=True)


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.types import interrupt


from app.config.payment_transaction_process_config import structured_evaluator_llm, _conversation_llm # Use your actual LLM config imports

_REQUIRED_KYC = ["full_name", "id_card_num", "phone_number", "nationality"]

# 1. Relaxed schema for robust extraction
class KYCExtraction(BaseModel):
    full_name: str | None = Field(default=None, description="The user's full name.")
    id_card_num: str | None = Field(default=None, description="The user's ID card number or passport number. Accept any string of letters/numbers.")
    phone_number: str | None = Field(default=None, description="The user's phone number.")
    nationality: str | None = Field(default=None, description="The user's nationality or country of origin.")

_extraction_llm = structured_evaluator_llm.with_structured_output(KYCExtraction)

def extract_kyc_details(state: KYCState) -> dict:
    
    current_data = {f: state.get(f) for f in _REQUIRED_KYC if state.get(f)}
    
    # 2. Feed the ENTIRE message history to maintain context
    extraction_messages = [
        SystemMessage(content=f"Extract KYC details from this conversation. Current verified data: {current_data}. Return null for missing fields.")
    ] + state.get("messages", [])
    
    extracted: KYCExtraction = _extraction_llm.invoke(extraction_messages)

    updates: dict = {}
    for field, value in extracted.model_dump().items():
        if value is not None and state.get(field) is None:
            updates[field] = value

    current = {**state, **updates}
    missing = [f for f in _REQUIRED_KYC if current.get(f) is None]

    # 3. Objective reality check in the terminal
    print(f"\n--- [DEBUG] KYC EXTRACTION ---")
    print(f"Extracted Data: {current}")
    print(f"Still Missing: {missing}")
    print(f"------------------------------\n")

    if missing:
        collected_summary = ", ".join(
            f"{f}={current[f]}" for f in _REQUIRED_KYC if current.get(f) is not None
        ) or "nothing yet"

        question_msg = _conversation_llm.invoke([
            SystemMessage(content=(
                "You are a strict but polite bank compliance officer collecting KYC (Know Your Customer) information. "
                "Ask the customer for the missing information naturally. Ask for ONE or TWO things at a time. "
                "Do NOT use technical field names. Do NOT say you have all the details."
            )),
            SystemMessage(content=f"Already collected: {collected_summary}. STRICTLY STILL NEEDED: {', '.join(missing)}."),
        ] + state.get("messages", []))

        question_text = question_msg.content
        user_reply = interrupt(question_text)

        updates["messages"] = [
            AIMessage(content=question_text),
            HumanMessage(content=user_reply),
        ]

    return updates

# 4. Routing logic mapped to our next database node
def route_after_extraction(state: KYCState) -> str:
    missing = [f for f in _REQUIRED_KYC if state.get(f) is None]
    
    if missing:
        return "extract_kyc_details"
        
    # Once we have all 4 fields, we hit the database
    return "upsert_kyc_record"

graph = StateGraph(KYCState)

graph.add_node("extract_kyc_details", extract_kyc_details)
graph.add_node("upsert_kyc_record", upsert_kyc_record) # <--- ADDED
graph.add_node("kyc_agent", kyc_agent)
graph.add_node("action", tool_node)
graph.add_node("kyc_agent_results", kyc_agent_results)
graph.add_node('human_review', human_review)
graph.add_node('notify_customer', notify_customer)


graph.add_edge(START, "extract_kyc_details")

graph.add_conditional_edges("extract_kyc_details", route_after_extraction)

graph.add_conditional_edges("upsert_kyc_record", route_after_upsert) # <--- ADDED
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




if __name__ == "__main__":
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
    config3 = {"configurable": {"thread_id": "kyc-test-3"}}
    result = workflow.invoke(initial_state, config=config3)
    print("Score:", result['kyc_score'])
    print("Status:", result['verification_status'])
    print("Notification:", result['notification_message'])
