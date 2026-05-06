from app.schemas.kyc_agent_schema import KYCState


def notify_customer(state: KYCState) -> dict:
    if state['kyc_score'] >= 0.8:
        message = f"Dear {state['full_name']}, your KYC is approved."
    elif state['human_decision'] == 'approve':
        message = f"Dear {state['full_name']}, your KYC is approved after manual review."
    else:
        message = f"Dear {state['full_name']}, your KYC is rejected. Reason: {state['rejection_reason']}"
    return {"notification_message": message, 
        "kyc_status": "completed"
    }