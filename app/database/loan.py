from sqlalchemy.orm import Session
from app.database.engine import engine
from app.models import Loan, User
from datetime import datetime


def create_loan_entry(state: dict) -> dict:
    with Session(engine) as session:
        new_loan = Loan(
            user_id=state.get("user_id", 1), # Default to 1 for testing
            full_name=state.get("full_name"),
            loan_amount=state.get("loan_amount"),
            loan_purpose=state.get("loan_purpose"),
            monthly_income=state.get("monthly_income"),
            loan_term_months=state.get("loan_term_months"),
            loan_status="pending"
        )
        session.add(new_loan)
        session.commit()
        session.refresh(new_loan)
        return {"loan_id": new_loan.loan_id}

def update_loan_final_status(state: dict):
    loan_id = state.get("loan_id")
    if not loan_id:
        print("[ERROR] No loan_id found in state. Cannot update database.")
        return

    with Session(engine) as session:
        loan_record = session.query(Loan).filter(Loan.loan_id == loan_id).first()
        if loan_record:
            loan_record.loan_status = state.get("loan_status")
            loan_record.notification_message = state.get("notification_message")
            loan_record.credit_score = state.get("credit_score")
            loan_record.verified_income_ratio = state.get("verified_income_ratio")
            loan_record.underwriting_decision = state.get("underwriting_decision")
            session.commit()
            print(f"[DB SUCCESS] Loan {loan_id} updated to {loan_record.loan_status}")