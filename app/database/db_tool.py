from sqlalchemy.orm import Session
from database.engine import engine
from models import Account
from models import Transaction
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