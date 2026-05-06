from app.schemas.kyc_agent_schema import KYCState

def score_condition(state: KYCState):
    if state["kyc_score"] >= 0.8:
        return 'approve'
    elif state['kyc_score'] >= 0.4: 
         return 'human_review'
    else:
        return 'reject'