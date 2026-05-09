
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

graph.add_node("planner", planner)
graph.add_node("credit_verification", credit_verification)
graph.add_node("income_verification", income_verification)
graph.add_node("underwriting_decision", underwriting_decision)
graph.add_node("human_review", human_review)
graph.add_node("notify_customer", notify_customer)

graph.add_edge(START, "planner")
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