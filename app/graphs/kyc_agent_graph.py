import os
from app.agents.kyc_agents.notify_customer import notify_customer
from app.agents.kyc_agents.human_review_kyc import human_review
from app.agents.kyc_agents.kyc_agent import kyc_agent
from app.agents.kyc_agents.kyc_agent_results import kyc_agent_results
from app.schemas.kyc_agent_schema import KYCState
from app.conditions.kyc_process_conditions.kyc_tool_conditions import tools_condition
from app.conditions.kyc_process_conditions.score_conditions import score_condition
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


# ── Extraction Schema ────────────────────────────────────────────────────────

class KYCExtraction(BaseModel):
    id_card_num:    Optional[str] = Field(None, description="National ID card number.")
    full_name:      Optional[str] = Field(None, description="Customer's full legal name exactly as on ID.")
    phone_number:   Optional[str] = Field(None, description="Phone number.")
    issue_date:     Optional[str] = Field(None, description="ID card issue date in YYYY-MM-DD format.")
    expiry_date:    Optional[str] = Field(None, description="ID card expiry date in YYYY-MM-DD format.")
    nationality:    Optional[str] = Field(None, description="Nationality.")
    date_of_birth:  Optional[str] = Field(None, description="Date of birth in YYYY-MM-DD format.")
    place_of_birth: Optional[str] = Field(None, description="City or place of birth.")
    gender:         Optional[str] = Field(None, description="Gender: Male or Female.")


_extraction_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY2"),  # structured extraction — KEY2
).with_structured_output(KYCExtraction)

_conversation_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY1"),  # user-facing conversation — KEY1
)

_REQUIRED = ["id_card_num", "full_name", "phone_number", "issue_date",
             "expiry_date", "nationality", "date_of_birth", "place_of_birth", "gender"]


# ── Extraction Node ──────────────────────────────────────────────────────────

def extract_kyc_details(state: KYCState) -> dict:
    """
    Reads the chat history, extracts KYC fields via structured LLM output.
    Interrupts with a natural question if any required field is still missing.
    """
    conversation = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.get("messages", [])
        if hasattr(m, "content")
    )

    extracted: KYCExtraction = _extraction_llm.invoke(
        f"Read this conversation and extract KYC identity verification details. "
        f"Return null for any field not mentioned.\n\n{conversation}"
    )

    updates: dict = {}
    for field, value in extracted.model_dump().items():
        if value is not None and not state.get(field):
            updates[field] = value

    current = {**state, **updates}
    missing = [f for f in _REQUIRED if not current.get(f)]

    if missing:
        collected_summary = ", ".join(
            f"{f}={current[f]}" for f in _REQUIRED if current.get(f)
        ) or "nothing yet"

        question_msg = _conversation_llm.invoke([
            SystemMessage(content=(
                "You are a bank compliance officer conducting a KYC (Know Your Customer) verification. "
                "Ask the customer for the next missing identity detail naturally and politely. "
                "Ask for one or two things at a time. Do not use technical field names."
            )),
            SystemMessage(content=f"Already collected: {collected_summary}. Still needed: {', '.join(missing)}."),
        ] + state.get("messages", []))

        question_text = question_msg.content
        user_reply = interrupt(question_text)

        reply_extracted: KYCExtraction = _extraction_llm.invoke(
            f"The officer asked: \"{question_text}\"\n"
            f"The user replied: \"{user_reply}\"\n"
            f"Extract any KYC identity fields from the reply. Return null for fields not mentioned."
        )

        for field, value in reply_extracted.model_dump().items():
            if value is not None and not current.get(field):
                updates[field] = value

        updates["messages"] = [
            AIMessage(content=question_text),
            HumanMessage(content=user_reply),
        ]

    return updates


def route_after_extraction(state: KYCState) -> str:
    missing = [f for f in _REQUIRED if not state.get(f)]
    return "kyc_agent" if not missing else "extract_kyc_details"


graph = StateGraph(KYCState)

graph.add_node("extract_kyc_details", extract_kyc_details)
graph.add_node("kyc_agent", kyc_agent)
graph.add_node("action", tool_node)
graph.add_node("kyc_agent_results", kyc_agent_results)
graph.add_node('human_review', human_review)
graph.add_node('notify_customer', notify_customer)


graph.add_edge(START, "extract_kyc_details")
graph.add_conditional_edges("extract_kyc_details", route_after_extraction)
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