import datetime
import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.schemas.payment_transaction_process_schema import TransferState, FraudCheck
from app.agents.payment_transaction_process_agents.balance_check_function import balance_check
from app.agents.payment_transaction_process_agents.fraud_check_function import fraud_check
from app.agents.payment_transaction_process_agents.notify_customer_functions import notify_customer
from app.agents.payment_transaction_process_agents.human_review import human_review
from app.agents.payment_transaction_process_agents.rejected_function import rejected
from app.conditions.payment_transaction_process_condition.process_transfer import process_transfer
from app.conditions.payment_transaction_process_condition.balance_check_condition import balance_check_condition
from app.conditions.payment_transaction_process_condition.fraud_check_condition import fraud_check_condition
from app.conditions.payment_transaction_process_condition.human_review_condition import human_review_condition

load_dotenv()


# ── Extraction Schema ────────────────────────────────────────────────────────

class TransferExtraction(BaseModel):
    sender_account_number: str | None = Field(default=None, description="The account sending money. Accept ANY string of characters or numbers.")
    receiver_account_number: str | None = Field(default=None, description="The account receiving money. Accept ANY string of characters or numbers.")
    money: float | None = Field(default=None, description="The numerical amount to transfer")

_extraction_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
).with_structured_output(TransferExtraction)

_conversation_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0.4,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)

from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import interrupt

# ONLY require the fields the user must provide in the chat
_REQUIRED = ["sender_account_number", "receiver_account_number", "money"]

# ── Extraction Node ──────────────────────────────────────────────────────────

def extract_transfer_details(state: TransferState) -> dict:
    
    current_data = {f: state.get(f) for f in _REQUIRED if state.get(f)}
    
    # 2. Feed the ENTIRE message history to the extractor to maintain context
    extraction_messages = [
        SystemMessage(content=f"Extract transfer details from this conversation. Current verified data: {current_data}. Return null for missing fields.")
    ] + state.get("messages", [])
    
    extracted: TransferExtraction = _extraction_llm.invoke(extraction_messages)

    updates: dict = {}
    for field, value in extracted.model_dump().items():
        if value is not None and state.get(field) is None:
            updates[field] = value

    current = {**state, **updates}
    missing = [f for f in _REQUIRED if current.get(f) is None]

    # 3. Objective reality check in the terminal
    print(f"\n--- [DEBUG] EXTRACTION LOGIC ---")
    print(f"Extracted Data: {current}")
    print(f"Still Missing: {missing}")
    print(f"--------------------------------\n")

    if missing:
        collected_summary = ", ".join(
            f"{f}={current[f]}" for f in _REQUIRED if current.get(f) is not None
        ) or "nothing yet"

        question_msg = _conversation_llm.invoke([
            SystemMessage(content=(
                "You are a helpful bank teller. "
                "Ask the customer for the missing information naturally. Ask for ONE thing at a time. "
                "CRITICAL: Do NOT say you have all the details. You are explicitly missing data."
            )),
            SystemMessage(content=f"Already collected: {collected_summary}. STRICTLY STILL NEEDED: {', '.join(missing)}."),
        ] + state.get("messages", []))

        question_text = question_msg.content
        user_reply = interrupt(question_text)

        # Save the interaction so the next loop has the full context
        updates["messages"] = [
            AIMessage(content=question_text),
            HumanMessage(content=user_reply),
        ]

    if not missing and not updates.get("sent_time") and not state.get("sent_time"):
        updates["sent_time"] = datetime.now().strftime("%H:%M") 

    return updates

def route_after_extraction(state: TransferState) -> str:
    missing = [f for f in _REQUIRED if state.get(f) is None]
    return "balance_check" if not missing else "extract_transfer_details"

transfer_workflow = StateGraph(TransferState)

#nodes
transfer_workflow.add_node("extract_transfer_details", extract_transfer_details)
transfer_workflow.add_node("balance_check", balance_check)
transfer_workflow.add_node("fraud_check", fraud_check)
transfer_workflow.add_node("notify_customer", notify_customer)
transfer_workflow.add_node("human_review", human_review)
transfer_workflow.add_node("process_transfer", process_transfer)
transfer_workflow.add_node("rejected", rejected)

#edges
transfer_workflow.add_edge(START, "extract_transfer_details")
transfer_workflow.add_conditional_edges("extract_transfer_details", route_after_extraction)
transfer_workflow.add_conditional_edges('balance_check', balance_check_condition)
transfer_workflow.add_conditional_edges('fraud_check', fraud_check_condition)
transfer_workflow.add_conditional_edges('human_review', human_review_condition)
transfer_workflow.add_edge('process_transfer', 'notify_customer')
transfer_workflow.add_edge('rejected', 'notify_customer')
transfer_workflow.add_edge('notify_customer', END)

transfer_workflow = transfer_workflow.compile(checkpointer=MemorySaver())



# #Test code
# initial_state = {
#     "user": "John Doe",
#     "account_num": 123456,
#     "to_transfer_acc_num": 999999,
#     "money": 15000,
#     "total_balance": 10000,
#     "sent_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "balance_check_status": "",
#     "fraud_score": 0.4,
#     "is_fraud": True,
#     "human_decision": "",
#     "transaction_status": "",
#     "notification_message": ""
# }

# config = {"configurable": {"thread_id": "test-3"}}

# result = transfer_workflow.invoke(initial_state, config=config)
# print(result)

# # # to approve
# # result = transfer_workflow.invoke(
# #     Command(resume="approve"),
# #     config=config
# # )
# # print(result)

# # to reject
# result = transfer_workflow.invoke(
#     Command(resume="reject"),
#     config=config
# )
# print(result)


if __name__ == "__main__":
    # test reject scenario
    initial_state = {
        "user": "John Doe",
        "account_num": 123456,
        "to_transfer_acc_num": 999999,
        "money": 9500,
        "total_balance": 10000,  # 95% of balance
        "sent_time": "2026-05-03 03:00:00",  # 3am
        "fraud_score": 0.0,
        "is_fraud": False,
        "human_decision": "",
        "transaction_status": "",
        "notification_message": "",
        "messages": [],
    }

    config = {"configurable": {"thread_id": "test-reject-1"}}

    # first invoke - will interrupt at human review
    result = transfer_workflow.invoke(initial_state, config=config)
    print("INTERRUPTED:", result)

    # resume with reject
    result = transfer_workflow.invoke(Command(resume="reject"), config=config)
    print("REJECTED:", result)