from app.schemas.loan_agent_schema import LoanState
from app.database.loan import update_loan_final_status

def notify_customer(state: LoanState) -> dict:
    decision = state.get("underwriting_decision") 
    tasks = state.get("tasks", [])
    full_name = state.get("full_name", "Valued Customer")
    
    if "ineligible" in tasks or not tasks:
        status = "rejected"
        message = f"Dear {full_name}, your application is ineligible for this loan type."
    elif decision == "auto_approve":
        status = "approved"
        message = f"Congratulations {full_name}, your loan has been automatically approved!"
    elif decision == "auto_decline":
        status = "rejected"
        message = f"Dear {full_name}, unfortunately your loan application has been declined."
    elif decision == "human_review":
        human_result = state.get("human_review_result")
        status = "approved" if human_result == "approved" else "rejected"
        message = f"Dear {full_name}, after manual review, your loan has been {status}."
    else:
        status = "error"
        message = "There was an error processing your application."

    state["loan_status"] = status
    state["rejection_reason"] = message if status == "rejected" else None

    update_loan_final_status(state)

    return {
        "loan_status": status,
        "notification_message": message,
        "workflow_status": "completed"
    }