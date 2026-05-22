from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database.engine import engine 
from app.models import User, KYC        
from app.agents.bank_manager.agent import master_graph
from langgraph.types import Command
from app.utils.deps import require_admin, UserContext

router = APIRouter()

class AdminReviewPayload(BaseModel):
    user_id: int
    action: str
    type: str = "kyc"
    item_id: str = ""

@router.get("/pending-kyc")
async def get_pending_kyc(admin: UserContext = Depends(require_admin)):
    """
    Reads pending KYC and Loan requests from the database for human review.
    """
    from app.models import Loan # Import Loan if not imported
    with Session(engine) as db:
        results = []
        
        # 1. Fetch pending KYC
        kycs = db.query(KYC).filter(KYC.verification_status.in_(["pending", "pending_review"])).all()
        for kyc in kycs:
            user = kyc.user
            results.append({
                "type": "KYC",
                "item_id": f"kyc_{kyc.kyc_id}",
                "user_id": user.user_id if user else 0,
                "full_name": user.full_name if user else "Unknown",
                "id_card_num": user.id_card_num if user else "Unknown",
                "loan_amount": 0,
                "loan_term_months": 0,
                "loan_purpose": "Identity Verification"
            })
            
        # 2. Fetch pending Loans
        loans = db.query(Loan).filter(Loan.underwriting_decision == "human_review", Loan.loan_status == "pending").all()
        for loan in loans:
            user = loan.borrower
            results.append({
                "type": "Loan",
                "item_id": f"loan_{loan.loan_id}",
                "user_id": user.user_id if user else 0,
                "full_name": loan.full_name,
                "id_card_num": user.id_card_num if user else "Unknown",
                "loan_amount": loan.loan_amount,
                "loan_term_months": loan.loan_term_months,
                "loan_purpose": loan.loan_purpose
            })
            
        return results


# --- 2. THE CRITICAL FIX: DYNAMIC GRAPH ROUTE FOR ZYAN ---
# Inside app/api/v1/admin.py
@router.post("/pending-review")
async def get_all_pending_reviews(admin: UserContext = Depends(require_admin)):
    results = []
    test_threads = ["999"] # Your testing thread ID
    
    for thread_id in test_threads:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = master_graph.get_state(config)
        
        if snapshot.next:
            state_values = snapshot.values
            
            # Map the exact keys your LangGraph state dictionary uses
            results.append({
                "user_id": int(thread_id) if thread_id.isdigit() else 999,
                "full_name": state_values.get("full_name", "Unknown AI User"),
                "id_card_num": thread_id,
                
                # Explicit Loan Data Variables
                "loan_amount": state_values.get("loan_amount", 0),
                "loan_term_months": state_values.get("loan_term_months", 0),
                "loan_purpose": state_values.get("loan_purpose", "Not specified")
            })
            
    return results


@router.post("/review")
async def review_transaction(payload: AdminReviewPayload, admin: UserContext = Depends(require_admin)):
    from app.models import Loan
    from app.graphs.loan_agent_graph import loan_app
    
    with Session(engine) as db:
        if payload.type == "KYC":
            # Process KYC Review
            kyc_id = int(payload.item_id.split("_")[1])
            kyc_record = db.query(KYC).filter(KYC.kyc_id == kyc_id).first()
            if kyc_record and kyc_record.user:
                # Resume the kyc_app graph instead of manually updating DB
                from app.graphs.kyc_agent_graph import kyc_app
                thread_id = kyc_record.user.id_card_num
                config = {"configurable": {"thread_id": thread_id}}
                
                resume_data = {"action": payload.action, "reason": "Admin Portal Review"}
                kyc_app.invoke(Command(resume=resume_data), config)
                
            return {
                "status": "success",
                "message": f"KYC for User {payload.user_id} has been permanently {payload.action}d.",
            }
            
        elif payload.type == "Loan":
            # Process Loan Review
            loan_id = int(payload.item_id.split("_")[1])
            loan_record = db.query(Loan).filter(Loan.loan_id == loan_id).first()
            if loan_record:
                loan_record.loan_status = "approved" if payload.action == "approve" else "rejected"
                db.commit()
                
                # Resume the LangGraph workflow using the saved thread_id
                thread_id = loan_record.notification_message
                if thread_id:
                    config = {"configurable": {"thread_id": thread_id}}
                    
                    # We resume the specific loan_app graph
                    # The value passed to resume is what interrupt() returns in human_review.py
                    decision = "approved" if payload.action == "approve" else "rejected"
                    loan_app.invoke(Command(resume=decision), config)
                    
            return {
                "status": "success",
                "message": f"Loan {loan_id} has been permanently {payload.action}d.",
            }

    return {"status": "failed", "message": "Unknown type or item not found."}