from fastapi import APIRouter, HTTPException, Body
from langgraph.types import Command
from sqlalchemy.orm import Session
from app.database.engine import engine
from app.models import User # Ensure path matches your structure
from app.graphs.kyc_agent_graph import kyc_app
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

@router.get("/status/{user_id}")
async def get_kyc_status(user_id: int):
    """
    Queries the DB to show the current KYC state to the user.
    """
    with Session(engine) as db:
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not user.kyc_record:
            return {
                "user_id": user_id,
                "full_name": user.full_name,
                "status": "no_record",
                "message": "Please submit your KYC documents to begin verification."
            }

        return {
            "user_id": user_id,
            "full_name": user.full_name,
            "verification_status": user.kyc_record.verification_status,
            "kyc_score": user.kyc_record.kyc_score,
            "last_updated": user.kyc_record.last_updated,
            "reject_reason": user.kyc_record.reject_reason if user.kyc_record.verification_status == "rejected" else None
        }


@router.post("/review")
async def review_kyc(
    id_card_num: str = Body(..., embed=True), 
    decision: str = Body(..., embed=True),
    reason: Optional[str] = Body(None, embed=True)
):
    """
    Resumes a paused KYC graph using a human decision.
    """
    config = {"configurable": {"thread_id": id_card_num}}
    
    # Check if the graph is actually waiting for review
    snapshot = kyc_app.get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="This application is not pending review.")

    # We use Command(resume=...) to pass the human decision into the graph
    # Your human_review node will receive this dict
    resume_data = {"action": decision, "reason": reason}
    
    # Resume execution
    final_state = kyc_app.invoke(Command(resume=resume_data), config)

    return {
        "status": final_state.get("verification_status"),
        "message": f"Review complete. Decision: {decision}"
    }

class KYCSubmission(BaseModel):
    full_name: str
    id_card_num: str
    nationality: str
    phone_number: str

@router.post("/submit")
async def submit_kyc(data: KYCSubmission):
    config = {"configurable": {"thread_id": data.id_card_num}}
    
    initial_state = {
        "full_name": data.full_name,
        "id_card_num": data.id_card_num,
        "nationality": data.nationality,
        "phone_number": data.phone_number,
        "messages": [] # Initialize messages if your graph expects them
    }
    
    print(f"--- [DEBUG] INVOKING KYC GRAPH FOR {data.id_card_num} ---")
    final_state = kyc_app.invoke(initial_state, config)
    print("--- [DEBUG] KYC GRAPH INVOKE FINISHED ---")
    
    snapshot = kyc_app.get_state(config)
    
    if snapshot.next: # If there are more nodes to run, it means we hit an interrupt
        return {
            "status": "pending_review",
            "message": "Application flagged for manual review.",
            "score": final_state.get("kyc_score"),
            "thread_id": data.id_card_num
        }

    return {
        "status": final_state.get("verification_status"),
        "message": final_state.get("notification_message") or final_state.get("reject_reason") or "KYC Processed Successfully",
        "score": final_state.get("kyc_score")
    }

@router.get("/pending-kyc")
async def get_pending_kyc():
    """
    Queries the SQL database for any users whose KYC status is currently 'pending_review'
    """
    with Session(engine) as db:
        # Query your SQL tables for pending records
        pending_records = db.query(User).join(User.kyc_record).filter(
            User.kyc_record.verification_status == "pending_review"
        ).all()
        
        results = []
        for user in pending_records:
            results.append({
                "user_id": user.user_id,
                "full_name": user.full_name,
                "nationality": user.kyc_record.nationality if user.kyc_record else "Unknown",
                "id_card_num": user.kyc_record.id_card_num if user.kyc_record else "N/A",
                "phone_number": user.kyc_record.phone_number if user.kyc_record else "N/A",
                "email": user.email if hasattr(user, 'email') else "N/A"
            })
        return results