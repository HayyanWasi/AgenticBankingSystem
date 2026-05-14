from schemas.loan_agent_schema import LoanState

def route_underwriting(state: LoanState):
    decision = state["underwriting_decision"]
    if decision == "auto_approve" or decision == "auto_decline":
        return "notify_customer"
    elif decision == "human_review":
        return "human_review"
    elif decision == "ineligible":
        return "notify_customer"