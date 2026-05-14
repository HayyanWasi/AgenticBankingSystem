import os
from agents.kyc_agents.notify_customer import notify_customer
from agents.kyc_agents.human_review_kyc import human_review
from agents.kyc_agents.kyc_agent import kyc_agent
from agents.kyc_agents.kyc_agent_results import kyc_agent_results
from schemas.kyc_agent_schema import KYCState
from conditions.kyc_process_conditions.kyc_tool_conditions import tools_condition
from conditions.kyc_process_conditions.score_conditions import score_condition
from agents.kyc_agents.upsert_kycrecord import upsert_kyc_record, route_after_upsert
from config.kyc_config import tool_node
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


from config.kyc_config import kyc_conversation_llm as _conversation_llm, kyc_extraction_llm_base

_REQUIRED_KYC = ["full_name", "id_card_num", "phone_number", "nationality"]

# 1. Relaxed schema for robust extraction
class KYCExtraction(BaseModel):
    full_name: str | None = Field(default=None, description="The user's full name.")
    id_card_num: str | None = Field(default=None, description="The user's ID card number or passport number. Accept any string of letters/numbers.")
    phone_number: str | None = Field(default=None, description="The user's phone number.")
    nationality: str | None = Field(default=None, description="The user's nationality or country of origin.")

_extraction_llm = kyc_extraction_llm_base.with_structured_output(KYCExtraction)

def extract_kyc_details(state: KYCState) -> dict:
    
    current_data = {f: state.get(f) for f in _REQUIRED_KYC if state.get(f)}
    
    # 2. Feed the ENTIRE message history to maintain context
    extraction_messages = [
        HumanMessage(content=f"Extract KYC details from this conversation. Current verified data: {current_data}. Return null for missing fields.")
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

if __name__ == "__main__":
    # 1. Setup the High-Risk Identity
    initial_state = {
        "id_card_num": "999-000-111",
        "full_name": "Musa Al Sadr",
        "phone_number": "+1-555-0199",
        "nationality": "North Korea", # High-risk country
        "kyc_score": 0.0,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": "audit-v1-test"}}

    # --- PHASE A: AI Extraction & Initial Scoring ---
    print("\n--- PHASE 1: AI SCORING & FLAG ---")
    result = workflow.invoke(initial_state, config=config)
    
    # Check if the graph correctly paused
    print(f"Current Status in State: {result.get('verification_status')}")
    print(f"AI Score: {result.get('kyc_score')}")

    # --- PHASE B: Human Review (Resuming) ---
    # We simulate an admin looking at the conversation history and approving
    print("\n--- PHASE 2: HUMAN OVERRIDE ---")
    
    # We pass the final decision. Note: human_review node will now call
    # update_kyc_decision including the messages list.
    resume_command = Command(resume={
    "action": "reject", 
    "reason": "ID card format is invalid and nationality is on the prohibited list."
})

    # final_result = workflow.invoke(resume_command, config=config)
    final_result = workflow.invoke(resume_command, config=config)

    print("\n--- TEST COMPLETE ---")
    print(f"Final Status: {final_result.get('verification_status')}")
    print(f"Notification: {final_result.get('notification_message')}")