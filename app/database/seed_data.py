from sqlalchemy.orm import Session
from database.engine import engine
from models import User, Account, Base

def seed_database():
    # Rebuild tables just in case they were deleted
    Base.metadata.create_all(bind=engine)
    
    with Session(engine) as session:
        # Check if data already exists to avoid duplicates
        if session.query(User).first():
            print("Database already contains data. Skipping seed.")
            return

        print("Seeding database with test accounts...")
        
        # Create Alice (Sender)
        alice = User(full_name="Alice Smith", id_card_num="ID-111", phone_number="555-0100", nationality="US")
        session.add(alice)
        session.commit() # Commit to generate her user_id
        
        alice_account = Account(account_number="ACCT-5555", user_id=alice.user_id, account_type="Checking", balance=500.0)
        session.add(alice_account)
        
        # Create Bob (Receiver)
        bob = User(full_name="Bob Jones", id_card_num="ID-222", phone_number="555-0200", nationality="UK")
        session.add(bob)
        session.commit() # Commit to generate his user_id
        
        bob_account = Account(account_number="ACCT-6666", user_id=bob.user_id, account_type="Savings", balance=100.0)
        session.add(bob_account)
        
        session.commit()
        print("Success! Created Alice (ACCT-5555, $500) and Bob (ACCT-6666, $100).")

if __name__ == "__main__":
    seed_database()