from datetime import datetime
from sqlalchemy.orm import Session
from database.engine import engine
from models import Account, Transaction, User, KYC

def get_account_balance(account_number: str) -> dict:
    """A strictly controlled tool for agents to check an account balance."""
    with Session(engine) as session:
        account = session.query(Account).filter(Account.account_number == account_number).first()
        
        if not account:
            return {"status": "failed", "reason": f"Account {account_number} does not exist."}
        
        return {
            "status": "success",
            "account_number": account.account_number,
            "balance": account.balance
        }

def transfer_funds(from_account_number, to_account_number, amount)-> dict:
    with Session(engine) as session:

        sender_account = session.query(Account).filter(Account.account_number == from_account_number).first()
        receiver_account = session.query(Account).filter(Account.account_number == to_account_number).first()

        if not sender_account:
            return {"status": "failed", "reason": "Sender account not found"}

        if not receiver_account:    
            return {"status": "failed", "reason": "Receiver account not found"}

        if amount< 0:
            return {"status": "failed", "reason": "amount can't be negative"}  

        if from_account_number == to_account_number:
            return {"status": "failed", "reason": "Sender and receiver accounts cannot be the same"}

        if sender_account.balance < amount:
            return {"status": "failed", "reason": "Insufficient funds"}

        

        # Execute Transfer
        sender_account.balance -= amount
        receiver_account.balance += amount

        transaction = Transaction(
        from_account_id=sender_account.account_id, 
        to_account_id=receiver_account.account_id,
        amount=amount
        )

        session.add(transaction)
        session.commit()

        return {
            "status": "success",
            "message": f"Successfully transferred {amount} from {from_account_number} to {to_account_number}",
            "amount": amount
        }
# if __name__ == "__main__":
#     print("Testing with a fake account...")
#     result = get_account_balance("FAKE-999")
#     print(result)


if __name__ == "__main__":
    print("--- Testing Database Tools ---")

    sender = "ACCT-5555"

    print(f"Initial balance for {sender}:", get_account_balance(sender))

    print("\nTest 1: Transfer to fake account")
    result1 = transfer_funds(sender, "FAKE-999", 50.0)
    print(result1)

    print("\nTest 2: Negative amount")
    result2 = transfer_funds(sender, "ACCT-5555", -10.0)
    print(result2)

    print("\nTest 3: Successful transfer")
    print(transfer_funds("ACCT-5555", "ACCT-6666", 150.0))



def get_kyc_status(full_name: str) -> dict:
    """Fetches the current KYC status for a user."""
    with Session(engine) as db:
        user = db.query(User).filter(User.full_name.ilike(full_name)).first()
        
        if not user:
            return {"status": "failed", "reason": "User not found in the system."}
            
        if not user.kyc_record:
            return {
                "status": "success", 
                "verification_status": "unverified",
                "kyc_score": 0.0,
                "message": "User exists but has no KYC record."
            }
            
        return {
            "status": "success",
            "verification_status": user.kyc_record.verification_status,
            "kyc_score": user.kyc_record.kyc_score,
            "reject_reason": user.kyc_record.reject_reason
        }

def upsert_user_kyc_details(full_name: str, id_card_num: str, phone_number: str, nationality: str) -> dict:
    """Updates user details and creates a pending KYC record if one doesn't exist."""
    with Session(engine) as db:
        # 1. Find or create the user
        user = db.query(User).filter(User.full_name.ilike(full_name)).first()
        
        if not user:
            # Check for duplicate ID cards to prevent SQLite Integrity Errors
            existing_id = db.query(User).filter(User.id_card_num == id_card_num).first()
            if existing_id:
                return {"status": "failed", "reason": "ID Card Number is already registered to another user."}
                
            user = User(
                full_name=full_name, 
                id_card_num=id_card_num, 
                phone_number=phone_number, 
                nationality=nationality
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            # Update existing user data
            user.id_card_num = id_card_num
            user.phone_number = phone_number
            user.nationality = nationality
            db.commit()

        # 2. Ensure a KYC record exists
        if not user.kyc_record:
            new_kyc = KYC(
                user_id=user.user_id, 
                kyc_score=0.0, 
                verification_status="pending"
            )
            db.add(new_kyc)
            db.commit()

        return {"status": "success", "message": "Identity details saved and KYC status set to pending."}

def update_kyc_decision(full_name: str, status: str, score: float, reject_reason: str = None) -> dict:
    """Saves the final compliance decision to the database."""
    with Session(engine) as db:
        user = db.query(User).filter(User.full_name.ilike(full_name)).first()
        
        if not user or not user.kyc_record:
            return {"status": "failed", "reason": "Cannot update decision. KYC record not found."}

        # Update the record
        user.kyc_record.verification_status = status
        user.kyc_record.kyc_score = score
        user.kyc_record.reject_reason = reject_reason
        user.kyc_record.last_updated = datetime.utcnow()

        db.commit()
        return {"status": "success", "message": f"KYC updated to '{status}'."}
        