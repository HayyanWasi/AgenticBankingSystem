from schemas.loan_agent_schema import LoanState

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
