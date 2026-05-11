import datetime
import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.schemas.payment_transaction_process_schema import TransferState, FruadCheck
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
    user:                Optional[str] = Field(None, description="The sender's full name.")
    account_num:         Optional[int] = Field(None, description="The sender's account number.")
    to_transfer_acc_num: Optional[int] = Field(None, description="The recipient's account number.")
    money:               Optional[int] = Field(None, description="The amount of money to transfer as a whole number.")
    total_balance:       Optional[int] = Field(None, description="The sender's current account balance.")


_extraction_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY2"),  # structured extraction — KEY2
).with_structured_output(TransferExtraction)

_conversation_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY1"),  # user-facing conversation — KEY1
)

_REQUIRED = ["user", "account_num", "to_transfer_acc_num", "money", "total_balance"]


# ── Extraction Node ──────────────────────────────────────────────────────────

def extract_transfer_details(state: TransferState) -> dict:
    """
    Reads the chat history, extracts transfer fields via structured LLM output.
    Interrupts with a natural question if any required field is still missing.
    """
    conversation = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.get("messages", [])
        if hasattr(m, "content")
    )

    extracted: TransferExtraction = _extraction_llm.invoke(
        f"Read this conversation and extract bank transfer details. "
        f"Return null for any field not mentioned.\n\n{conversation}"
    )

    updates: dict = {}
    for field, value in extracted.model_dump().items():
        if value is not None and state.get(field) is None:
            updates[field] = value

    current = {**state, **updates}
    missing = [f for f in _REQUIRED if current.get(f) is None]

    if missing:
        collected_summary = ", ".join(
            f"{f}={current[f]}" for f in _REQUIRED if current.get(f) is not None
        ) or "nothing yet"

        question_msg = _conversation_llm.invoke([
            SystemMessage(content=(
                "You are a helpful bank teller processing a money transfer. "
                "Ask the customer for the next missing piece of information naturally. "
                "Ask for one or two things at a time. Be friendly and concise."
            )),
            SystemMessage(content=f"Already collected: {collected_summary}. Still needed: {', '.join(missing)}."),
        ] + state.get("messages", []))

        question_text = question_msg.content
        user_reply = interrupt(question_text)

        reply_extracted: TransferExtraction = _extraction_llm.invoke(
            f"The teller asked: \"{question_text}\"\n"
            f"The user replied: \"{user_reply}\"\n"
            f"Extract any transfer fields from the reply. Return null for fields not mentioned."
        )

        for field, value in reply_extracted.model_dump().items():
            if value is not None and current.get(field) is None:
                updates[field] = value

        updates["messages"] = [
            AIMessage(content=question_text),
            HumanMessage(content=user_reply),
        ]

    # Set sent_time if not already present
    if not updates.get("sent_time") and not state.get("sent_time"):
        updates["sent_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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