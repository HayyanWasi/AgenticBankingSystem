
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














from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

graph = StateGraph(LoanState)

graph.add_node("planner", planner)
graph.add_node("credit_verification", credit_verification)
graph.add_node("income_verification", income_verification)

# 3. The Dynamic Router
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
        return [] 
    return [Send(task, state) for task in tasks]
    # The Send object is an instruction package. It takes two arguments: a destination (the node name) and a
    #  payload (the state).
    # This line loops through the tasks list.
graph.add_edge(START, "planner")

graph.add_conditional_edges("planner", route_tasks) 
# his connects the traffic controller to the graph. It tells the system: "When the planner finishes,
#  do not automatically move to the next node. Instead, pass the state to the route_tasks function and 
# execute whatever paths it returns."







def test_planner():
    # Mock input 1: Personal loan under 50000
    state1: LoanState = {
        "full_name": "John Doe",
        "id_card_num": "12345-6789012-3",
        "loan_amount": 30000.0,
        "loan_term_months": 12,
        "monthly_income": 50,
        "loan_purpose": "mortgage",
        "loan_reason": "Debt consolidation",
        "credit_score": 0,
        "credit_flags": [],
        "verified_income_ratio": 0.0,
        "risk_tier": "low",
        "underwriting_decision": "human_review",
        "human_review_result": None,
        "tasks": [],
        "workflow_status": "pending",
        "notification_message": "",
        "loan_status": ""
    }
    

    result1 = planner(state1)
    print(f"Resulting Tasks: {result1['tasks']}")
    print("-" * 20)

    # # Mock input 2: Mortgage
    # state2: LoanState = state1.copy()
    # state2["loan_purpose"] = "mortgage"
    # state2["loan_amount"] = 200000.0
    
    # print("Testing Case 2: Mortgage")
    # result2 = planner(state2)
    # print(f"Resulting Tasks: {result2['tasks']}")
    # print("-" * 20)

    # # Mock input 3: Business
    # state3: LoanState = state1.copy()
    # state3["loan_purpose"] = "business"
    # state3["loan_amount"] = 100000.0
    
    # print("Testing Case 3: Business")
    # result3 = planner(state3)
    # print(f"Resulting Tasks: {result3['tasks']}")
    # print("-" * 20)

if __name__ == "__main__":
    test_planner()

