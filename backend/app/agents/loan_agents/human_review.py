from app.schemas.loan_agent_schema import LoanState
from langgraph.types import interrupt
from langchain_core.runnables.config import RunnableConfig
from app.database.engine import engine
from sqlalchemy.orm import Session
from app.models import Loan, User

def human_review(state: LoanState, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    
    # Save the pending loan to the database so the Admin Portal can see it
    with Session(engine) as db:
        user = db.query(User).filter(User.id_card_num == state.get("id_card_num")).first()
        if user:
            # Check if this loan already exists (e.g. graph was re-run)
            existing = db.query(Loan).filter(Loan.user_id == user.user_id, Loan.underwriting_decision == "human_review").first()
            if not existing:
                new_loan = Loan(
                    user_id=user.user_id,
                    full_name=state.get("full_name"),
                    loan_amount=state.get("loan_amount"),
                    loan_purpose=state.get("loan_purpose"),
                    monthly_income=state.get("monthly_income"),
                    loan_term_months=state.get("loan_term_months"),
                    credit_score=state.get("credit_score"),
                    verified_income_ratio=state.get("verified_income_ratio"),
                    underwriting_decision="human_review",
                    loan_status="pending",
                    notification_message=thread_id  # Store thread_id here temporarily
                )
                db.add(new_loan)
                db.commit()

    decision = interrupt({
        "full_name": state.get("full_name"),
        "id_card_num": state.get("id_card_num"),
        "loan_amount": state.get("loan_amount"),
        "loan_term_months": state.get("loan_term_months"),
        "loan_purpose": state.get("loan_purpose"),
        "loan_reason": state.get("loan_reason"),
        "credit_score": state.get("credit_score"),
        "credit_flags": state.get("credit_flags"),
        "verified_income_ratio": state.get("verified_income_ratio"),
        "underwriting_decision": state.get("underwriting_decision")
    })
    return {"human_review_result": decision}