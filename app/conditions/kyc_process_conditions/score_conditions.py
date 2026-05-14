from schemas.kyc_agent_schema import KYCState

def score_condition(state: KYCState) -> str:
    status = state.get("verification_status")

    if status == "approved":
        return "approve"
    elif status == "human_review":
        return "human_review"
    else:
        return "reject"