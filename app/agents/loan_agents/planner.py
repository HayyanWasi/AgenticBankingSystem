from schemas.loan_agent_schema import LoanState
from database.loan import create_loan_entry


def planner(state: LoanState) -> dict:
    loan_purpose = state.get('loan_purpose')
    
    db_result = create_loan_entry(state)
    new_id = db_result.get("loan_id")

    if loan_purpose == "personal":
        tasks = ["credit_verification", "income_verification"]
    elif loan_purpose in ["business", "mortgage"]:
        tasks = ["credit_verification", "income_verification", "collateral_evaluation"]
    else:
        tasks = ["ineligible"]

    return {
        "tasks": tasks, 
        "loan_id": new_id
    }