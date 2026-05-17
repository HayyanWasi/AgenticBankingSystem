from app.schemas.loan_agent_schema import LoanState
from langgraph.types import interrupt


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