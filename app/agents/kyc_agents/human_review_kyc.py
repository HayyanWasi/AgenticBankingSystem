from app.schemas.kyc_agent_schema import KYCState
from langgraph.types import interrupt
from app.database.db_tool import update_kyc_decision



def human_review(state: KYCState) -> dict:
    # The pause
    decision_payload = interrupt({
        "question": f"Review required for {state.get('full_name')}",
        "current_score": state.get("kyc_score")
    })

    # The resume logic
    action = decision_payload.get("action")
    reason = decision_payload.get("reason", "Manual review completed")

    final_status = "approved" if action == "approve" else "rejected"

    id_card = state.get("id_card_num")
    history = state.get("messages", [])

    # Persist the decision to the database
    update_kyc_decision(
        id_card_num=id_card,
        status="approved" if action == "approve" else "rejected",
        score=state.get("kyc_score", 0.0),
        messages=history, # This is what populates the audit_trail
        reject_reason=reason
    )

    return {
        "verification_status": final_status,
        "human_decision": action,
        "reject_reason": reason
    }