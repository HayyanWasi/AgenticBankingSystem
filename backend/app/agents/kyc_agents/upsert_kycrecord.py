from app.schemas.kyc_agent_schema import KYCState
from app.database.db_tool import upsert_user_kyc_details



from app.database.db_tool import upsert_user_kyc_details

def upsert_kyc_record(state: KYCState) -> dict:
    """Bridges the Graph State to the SQLite Database."""
    result = upsert_user_kyc_details(
        full_name=state.get("full_name"),
        id_card_num=state.get("id_card_num"),
        phone_number=state.get("phone_number"),
        nationality=state.get("nationality")
    )

    if result["status"] == "success":
        return {
            "kyc_status": "success",
            "verification_status": "pending"
        }
    else:
        # Handle database conflicts (e.g., Duplicate ID)
        return {
            "kyc_status": "failed",
            "verification_status": "rejected",
            "reject_reason": result.get("reason")
        }

def route_after_upsert(state: KYCState) -> str:
    status = state.get("kyc_status")

    if status == "success":
        return "kyc_agent"
    elif status == "failed":
        # Route directly to customer notification to end the graph cleanly
        return "notify_customer"

    return "human_review"