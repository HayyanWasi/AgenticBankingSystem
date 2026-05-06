from app.schemas.kyc_agent_schema import KYCState
from langgraph.types import interrupt



def human_review(state: KYCState) -> dict:
    decision = interrupt({
        'full_name': state['full_name'],
        'id_card_num': state['id_card_num'],
        'kyc_score': state['kyc_score'],
        'verification_status': state['verification_status'],    
        'question': 'Approve or reject this KYC?'
    })
    return {'human_decision': decision}
