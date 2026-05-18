from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.engine import engine
from app.models import User, Account, Transaction, KYC
from sqlalchemy.orm import sessionmaker

router = APIRouter()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/dashboard/{user_id}")
def get_user_dashboard(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Get primary account balance (or user balance if you prefer)
    account = db.query(Account).filter(Account.user_id == user_id).first()
    balance = account.balance if account else user.balance

    # Get KYC Status
    kyc = db.query(KYC).filter(KYC.user_id == user_id).first()
    kyc_status = kyc.verification_status if kyc else "Not Started"

    # Get recent transactions (both sent and received)
    if account:
        sent = db.query(Transaction).filter(Transaction.from_account_id == account.account_id).all()
        received = db.query(Transaction).filter(Transaction.to_account_id == account.account_id).all()
        all_tx = sent + received
        all_tx.sort(key=lambda x: x.timestamp, reverse=True)
        recent_tx = all_tx[:5]
        
        # Format transactions
        formatted_tx = []
        for tx in recent_tx:
            is_sent = tx.from_account_id == account.account_id
            amount_str = f"-${tx.amount:,.2f}" if is_sent else f"+${tx.amount:,.2f}"
            
            # Simple icon mapping
            icon = "payments" if is_sent else "account_balance"
            type_str = "Transfer Out" if is_sent else "Transfer In"
            
            # Status styling
            status_color = "text-secondary" if tx.status == "success" else "text-on-surface-variant"
            status_icon = "check" if tx.status == "success" else "schedule"
            if tx.status == "pending_review":
                status_color = "text-error"
                status_icon = "warning"
                
            formatted_tx.append({
                "icon": icon,
                "name": f"Account {tx.to_account_id if is_sent else tx.from_account_id}",
                "type": type_str,
                "amount": amount_str,
                "status": (tx.status or "Unknown").title().replace("_", " "),
                "statusIcon": status_icon,
                "statusColor": status_color
            })
    else:
        formatted_tx = []

    return {
        "full_name": user.full_name,
        "balance": balance,
        "kyc_status": kyc_status,
        "recent_transactions": formatted_tx
    }
