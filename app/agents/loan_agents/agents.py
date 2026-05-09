
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Send, Command
from langgraph.graph import StateGraph, START, END
from app.schemas.loan_agent_schema import LoanState




def planner(state: LoanState)-> dict:
    loan_purpose = state['loan_purpose']
    loan_amount = state['loan_amount']
    
    if loan_purpose == "personal":
        tasks = ["credit_verification", "income_verification"]
        return {"tasks": tasks}
    elif loan_purpose == "business":
        tasks = ["credit_verification", "income_verification", "collateral_evaluation"]
        return {"tasks": tasks}
    elif loan_purpose == "mortgage":
        tasks = ["credit_verification", "income_verification", "collateral_evaluation"]
        return {"tasks": tasks}
    else:
        return {"tasks": ["ineligible"]} 
    



def credit_verification(state: LoanState) -> dict:
    monthly_income = state['monthly_income']
    loan_amount = state['loan_amount']
    if monthly_income > 50000:
        score = 750
    elif monthly_income >= 30000:
        score = 650
    else:
        score = 550


    flags = []
    if monthly_income < 30000:
        flags.append("low_income")
    if loan_amount > (10 * monthly_income):
        flags.append("high_debt_ratio")

    return {
        "credit_score": score,
        "credit_flags": flags
    }



def income_verification(state: LoanState)->dict:
    loan_amount = state['loan_amount']
    monthly_income = state['monthly_income']
    loan_term_months = state["loan_term_months"]

    verified_income_ratio = (loan_amount / loan_term_months) / monthly_income

    return {"verified_income_ratio": verified_income_ratio}

def underwriting_decision(state: LoanState) -> dict:
    credit_score = state["credit_score"]
    verified_income_ratio = state["verified_income_ratio"]
    credit_flags = state["credit_flags"]

    if credit_score > 700 and verified_income_ratio < 0.3:
        risk_tier = "low"
    elif credit_score < 600 or verified_income_ratio > 0.5:
        risk_tier = "high"
    else:
        risk_tier = "medium"

    if risk_tier == "low" and not credit_flags:
        decision = "auto_approve"
    elif risk_tier == "high" or "high_debt_ratio" in credit_flags or "low_income" in credit_flags:
        decision = "auto_decline"
    else:
        decision = "human_review" 

    return {"underwriting_decision": decision}




def human_review(state: LoanState) -> dict:
    decision = interrupt({
        "full_name": state["full_name"],
        "id_card_num": state["id_card_num"],
        "loan_amount": state["loan_amount"],
        "loan_term_months": state["loan_term_months"],
        "loan_purpose": state["loan_purpose"],
        "loan_reason": state["loan_reason"],
        "credit_score": state["credit_score"],
        "credit_flags": state["credit_flags"],
        "verified_income_ratio": state["verified_income_ratio"],
        "underwriting_decision": state["underwriting_decision"]
    })
    return {"human_review_result": decision}


def notify_customer(state: LoanState) -> dict:
    # Use .get() so it doesn't crash if the key doesn't exist yet
    decision = state.get("underwriting_decision") 
    tasks = state.get("tasks", [])
    full_name = state["full_name"]
    
    # Check for ineligibility FIRST
    if "ineligible" in tasks:
        status = "rejected"
        message = f"Dear {full_name}, your application is ineligible for this loan type."
        
    elif decision == "auto_approve":
        status = "approved"
        message = f"Congratulations {full_name}, your loan has been automatically approved!"
        
    elif decision == "auto_decline":
        status = "rejected"
        message = f"Dear {full_name}, unfortunately your loan application has been declined based on our automated risk assessment."
        
    elif decision == "human_review":
        human_result = state.get("human_review_result")
        if human_result == "approved":
            status = "approved"
            message = f"Congratulations {full_name}, after manual review, your loan has been approved!"
        else:
            status = "rejected"
            message = f"Dear {full_name}, after manual review, your loan application has been declined."
            
    else:
        status = "error"
        message = "There was an error processing your loan application."

    return {
        "loan_status": status,
        "notification_message": message,
        "workflow_status": "completed"
    }  

def route_underwriting(state: LoanState):
    decision = state["underwriting_decision"]
    if decision == "auto_approve" or decision == "auto_decline":
        return "notify_customer"
    elif decision == "human_review":
        return "human_review"
    elif decision == "ineligible":
        return "notify_customer"
    

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
        print(f"🚀 RUNNING SCENARIO: {scenario_name}")
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
            print(f"✅ Executed Node: {node_name}")
            
            # If we hit the notification node, print the final result
            if node_name == "notify_customer":
                print(f"\n🎯 FINAL RESULT: {event['notify_customer']['loan_status'].upper()}")
                print(f"📝 MESSAGE: {event['notify_customer']['notification_message']}")

        # Check if the graph paused for Human Review
        current_state = loan_graph.get_state(config)
        if current_state.next and "human_review" in current_state.next:
            print("\n⏸️ GRAPH PAUSED FOR HUMAN REVIEW")
            print("▶️ Simulating Human Approval...")
            for event in loan_graph.stream(Command(resume="approved"), config, stream_mode="updates"):
                if "notify_customer" in event:
                    print(f"\n🎯 FINAL RESULT: {event['notify_customer']['loan_status'].upper()}")
                    print(f"📝 MESSAGE: {event['notify_customer']['notification_message']}")

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