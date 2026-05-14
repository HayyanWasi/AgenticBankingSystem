from schemas.kyc_agent_schema import KYCState

# def notify_customer(state: KYCState) -> dict:
    
#     status = state.get("verification_status")
#     name = state.get("full_name", "Customer")
#     reason = state.get("reject_reason") or state.get("rejection_reason") or "Criteria not met."

    
#     if status == "approved":
#         message = f"Dear {name}, we are pleased to inform you that your KYC verification is successful. Your account is now active."
#     elif status == "rejected":
#         message = f"Dear {name}, unfortunately, we could not process your KYC at this time. Reason: {reason}"
#     else:
        
#         message = f"Dear {name}, your application is currently being processed. We will notify you once a decision is made."

#     print(f"--- [DEBUG] NOTIFICATION SENT ---")
#     print(f"Message: {message}")

#     return {
#         "notification_message": message,
#         "kyc_status": "completed"
#     }


def notify_customer(state: KYCState) -> dict:
    status = state.get("verification_status") # This will now be 'approved' or 'rejected'
    
    if status == "approved":
        msg = "Your KYC has been approved."
    elif status == "rejected":
        msg = f"Your KYC was rejected. Reason: {state.get('reject_reason')}"
    else:
        msg = "Your application is under review." # The fallback you saw
        
    return {"notification_message": msg}