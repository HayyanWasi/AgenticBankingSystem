from app.schemas.loan_agent_schema import LoanState

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
