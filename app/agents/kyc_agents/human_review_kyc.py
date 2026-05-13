from schemas.kyc_agent_schema import KYCState
from langgraph.types import interrupt
from database.db_tool import update_kyc_decision



def human_review(state: KYCState) -> dict:
    # 1. The Breakpoint
    decision_payload = interrupt({
        'full_name': state.get('full_name'),
        'id_card_num': state.get('id_card_num'),
        'kyc_score': state.get('kyc_score'),
        'verification_status': state.get('verification_status'),    
        'question': 'Approve or reject this KYC?'
    })

    # Extract decision from the admin input
    action = decision_payload.get('action')
    reason = decision_payload.get('reason')

    if action == "approve":
        final_status = "approved"
        kyc_signal = "success"
    else:
        final_status = "rejected"
        kyc_signal = "failed"

    # 3. Database Update
    update_kyc_decision(
        id_card_num=state.get('id_card_num'), 
        status=final_status,
        score=state.get('kyc_score', 0.0),
        reject_reason=reason
    )

    return {
        "verification_status": final_status,
        "kyc_status": kyc_signal,
        "reject_reason": reason
    }