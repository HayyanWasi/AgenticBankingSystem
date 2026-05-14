from schemas.loan_agent_schema import LoanState

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