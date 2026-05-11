
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Send, Command
from langgraph.graph import StateGraph, START, END
from app.schemas.loan_agent_schema import LoanState

from app.agents.loan_agents.planner import planner
from app.agents.loan_agents.credit_verification import credit_verification
from app.agents.loan_agents.income_verification import income_verification
from app.agents.loan_agents.underwriting_decision import underwriting_decision
from app.agents.loan_agents.human_review import human_review
from app.agents.loan_agents.notify_customer import notify_customer
from app.conditions.loan_agent_condition.underwriting_condition import route_underwriting

import os
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv(override=True)


# ── Extraction Schema ────────────────────────────────────────────────────────

class LoanExtraction(BaseModel):
    # Only the fields the user provides.
    # All are Optional so the LLM doesn't hallucinate missing data.
    full_name:        Optional[str]                                    = Field(None, description="The user's full legal name.")
    id_card_num:      Optional[str]                                    = Field(None, description="The user's national ID card number.")
    loan_amount:      Optional[float]                                  = Field(None, description="The total loan amount as a number.")
    loan_term_months: Optional[int]                                    = Field(None, description="Repayment duration in months.")
    monthly_income:   Optional[float]                                  = Field(None, description="The user's monthly income as a number.")
    loan_purpose:     Optional[Literal['personal', 'business', 'mortgage']] = Field(None, description="Type of loan.")
    loan_reason:      Optional[str]                                    = Field(None, description="Why the user wants the loan.")


_extraction_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY2"),  # structured extraction — KEY2
).with_structured_output(LoanExtraction)

_conversation_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY1"),  # user-facing conversation — KEY1
)

_REQUIRED = ["full_name", "id_card_num", "loan_amount", "loan_term_months", "monthly_income", "loan_purpose"]


# ── Extraction Node ──────────────────────────────────────────────────────────

def extract_loan_details(state: LoanState) -> dict:
    """
    Reads the full chat history and extracts loan fields using a structured LLM.
    If any required fields are still missing after extraction, generates a natural
    conversational question and interrupts to wait for the user's reply.
    """
    # Build a readable conversation string for the extraction LLM
    conversation = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.get("messages", [])
        if hasattr(m, "content")
    )

    # Extract whatever is present in the conversation
    extracted: LoanExtraction = _extraction_llm.invoke(
        f"Read this conversation and extract loan application details. "
        f"Return null for any field not mentioned.\n\n{conversation}"
    )

    # Merge into state — don't overwrite fields already collected
    updates: dict = {}
    for field, value in extracted.model_dump().items():
        if value is not None and not state.get(field):
            updates[field] = value

    # Check what's still missing after this extraction pass
    current = {**state, **updates}
    missing = [f for f in _REQUIRED if not current.get(f)]

    if missing:
        # Ask the user naturally for the missing fields
        collected_summary = ", ".join(
            f"{f}={current[f]}" for f in _REQUIRED if current.get(f)
        ) or "nothing yet"

        question_msg = _conversation_llm.invoke([
            SystemMessage(content=(
                "You are a friendly bank loan officer collecting a loan application. "
                "Ask the customer for the next missing piece of information naturally. "
                "Ask for one or two fields at a time — never dump all questions at once. "
                "Do not use field names; phrase the question conversationally."
            )),
            SystemMessage(content=f"Already collected: {collected_summary}. Still needed: {', '.join(missing)}."),
        ] + state.get("messages", []))

        question_text = question_msg.content

        # Pause and wait for user reply
        user_reply = interrupt(question_text)

        # Extract fields from the user's new reply
        reply_extracted: LoanExtraction = _extraction_llm.invoke(
            f"The loan officer asked: \"{question_text}\"\n"
            f"The user replied: \"{user_reply}\"\n"
            f"Extract any loan fields from the reply. Return null for fields not mentioned."
        )

        for field, value in reply_extracted.model_dump().items():
            if value is not None and not current.get(field):
                updates[field] = value

        updates["messages"] = [
            AIMessage(content=question_text),
            HumanMessage(content=user_reply),
        ]

    return updates


def route_after_extraction(state: LoanState) -> str:
    missing = [f for f in _REQUIRED if not state.get(f)]
    return "planner" if not missing else "extract_loan_details"


# The Dynamic Router
def route_tasks(state: LoanState):
    """
    Reads the 'tasks' list from the state and dynamically dispatches 
    execution to only the nodes listed.
    """
    tasks = state.get('tasks', [])
    # It extracts the specific list of tasks that the planner just decided on. If the user asked for a 
    # small personal loan, this variable now holds ["credit_verification","income_verification"]
    if "ineligible" in tasks or not tasks:
        
        # This is your kill switch. If the planner decided the user is ineligible,
        #  or if the list is empty, returning an empty list [] tells LangGraph that there are no nodes to route to.
        #  The execution stops here.
        return 'notify_customer' 
    return [Send(task, state) for task in tasks]

graph = StateGraph(LoanState)

graph.add_node("extract_loan_details", extract_loan_details)
graph.add_node("planner", planner)
graph.add_node("credit_verification", credit_verification)
graph.add_node("income_verification", income_verification)
graph.add_node("underwriting_decision", underwriting_decision)
graph.add_node("human_review", human_review)
graph.add_node("notify_customer", notify_customer)

graph.add_edge(START, "extract_loan_details")
graph.add_conditional_edges("extract_loan_details", route_after_extraction)
graph.add_conditional_edges("planner", route_tasks)


graph.add_edge("credit_verification", "underwriting_decision")
graph.add_edge("income_verification", "underwriting_decision")


graph.add_conditional_edges(
    "underwriting_decision",
    route_underwriting,
    {
        "notify_customer": "notify_customer",
        "human_review": "human_review"
    }
)


graph.add_edge("human_review", "notify_customer")
graph.add_edge("notify_customer", END)


memory = MemorySaver()
loan_graph = graph.compile(checkpointer=memory)


#testing 

# if __name__ == "__main__":
#     import uuid
    
#     # Create a unique thread ID for the MemorySaver checkpointer
#     thread_id = str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}

#     # The exact test state
#     initial_state = {
#         "full_name": "John Doe",
#         "id_card_num": "12345-6789012-3",
#         "loan_amount": 40000.0,
#         "loan_term_months": 12,
#         "monthly_income": 35000.0,
#         "loan_purpose": "personal",
#         "loan_reason": "Home improvement",
#         "credit_score": 0,
#         "credit_flags": [],
#         "verified_income_ratio": 0.0,
#         "tasks": []
#     }

#     print("--- STARTING GRAPH EXECUTION ---")
    
#     # 1. Run the graph until it hits the interrupt
#     for event in loan_graph.stream(initial_state, config, stream_mode="updates"):
#         print(event)
        
#     print("\n--- GRAPH PAUSED FOR HUMAN REVIEW ---")
    
#     # 2. Provide the human input to resume the graph
#     print("\n--- RESUMING WITH HUMAN APPROVAL ---")
#     # We pass the human decision back into the graph as if the human clicked 'Approve'
#     for event in loan_graph.stream(Command(resume="approved"), config, stream_mode="updates"):
#         print(event)


if __name__ == "__main__":
    import uuid
    from langgraph.types import Command

    def run_test_scenario(scenario_name: str, state_override: dict):
        print(f"\n{'='*60}")
        print(f"[START] RUNNING SCENARIO: {scenario_name}")
        print(f"{'='*60}")
        
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Base state template
        base_state = {
            "full_name": "John Doe",
            "id_card_num": "12345-6789012-3",
            "loan_amount": 0.0,
            "loan_term_months": 12,
            "monthly_income": 0.0,
            "loan_purpose": "personal",
            "loan_reason": "Testing",
            "credit_score": 0,
            "credit_flags": [],
            "verified_income_ratio": 0.0,
            "tasks": []
        }
        # Merge overrides into base state
        test_state = {**base_state, **state_override}

        # Run the graph
        for event in loan_graph.stream(test_state, config, stream_mode="updates"):
            # Print only the node name to keep the terminal clean
            node_name = list(event.keys())[0]
            print(f"[OK] Executed Node: {node_name}")
            
            # If we hit the notification node, print the final result
            if node_name == "notify_customer":
                print(f"\n[RESULT] FINAL RESULT: {event['notify_customer']['loan_status'].upper()}")
                print(f"[INFO] MESSAGE: {event['notify_customer']['notification_message']}")

        # Check if the graph paused for Human Review
        current_state = loan_graph.get_state(config)
        if current_state.next and "human_review" in current_state.next:
            print("\n[PAUSED] GRAPH PAUSED FOR HUMAN REVIEW")
            print("[RESUME] Simulating Human Approval...")
            for event in loan_graph.stream(Command(resume="approved"), config, stream_mode="updates"):
                if "notify_customer" in event:
                    print(f"\n[RESULT] FINAL RESULT: {event['notify_customer']['loan_status'].upper()}")
                    print(f"[INFO] MESSAGE: {event['notify_customer']['notification_message']}")

    # --- THE TEST CASES ---
    
    # 1. AUTO-APPROVE: > 50k income (750 score), small loan ratio.
    run_test_scenario("Auto-Approve", {
        "monthly_income": 60000.0, 
        "loan_amount": 10000.0,
        "loan_purpose": "personal"
    })

    # 2. AUTO-DECLINE: < 30k income (550 score + low_income flag), huge loan.
    run_test_scenario("Auto-Decline", {
        "monthly_income": 20000.0, 
        "loan_amount": 250000.0,
        "loan_purpose": "personal"
    })

    # 3. HUMAN REVIEW: Medium income (650 score), moderate ratio.
    run_test_scenario("Human Review", {
        "monthly_income": 40000.0, 
        "loan_amount": 40000.0,
        "loan_purpose": "personal"
    })

    # 4. INELIGIBLE: Unsupported loan purpose
    run_test_scenario("Ineligible Bypass", {
        "monthly_income": 50000.0, 
        "loan_amount": 10000.0,
        "loan_purpose": "student_loan" # Planner should flag this as ineligible
    })