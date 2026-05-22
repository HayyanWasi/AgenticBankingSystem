from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import random
from app.database.engine import engine
from app.models import User, Account
from app.utils.auth import get_password_hash, verify_password, create_access_token

router = APIRouter()

class RegisterPayload(BaseModel):
    email: str
    password: str
    full_name: str
    id_card_num: str

class LoginPayload(BaseModel):
    email: str
    password: str

@router.post("/register")
async def register_user(payload: RegisterPayload):
    with Session(engine) as db:
        # Check for existing email or ID card conflicts
        if db.query(User).filter(User.email == payload.email).first():
            raise HTTPException(status_code=400, detail="Email already registered.")
        if db.query(User).filter(User.id_card_num == payload.id_card_num).first():
            raise HTTPException(status_code=400, detail="ID Card already registered.")

        # Hash the password and save the record
        hashed_pw = get_password_hash(payload.password)
        new_user = User(
            email=payload.email,
            password_hash=hashed_pw,
            full_name=payload.full_name,
            id_card_num=payload.id_card_num
        )
        db.add(new_user)
        db.commit()
        
        return {"status": "success", "message": "Account created."}

@router.post("/login")
async def login_user(payload: LoginPayload):
    with Session(engine) as db:
        user = db.query(User).filter(User.email == payload.email).first()
        
        # Verify user exists and password matches
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        # Ensure the user has an account assigned
        account = db.query(Account).filter(Account.user_id == user.user_id).first()
        if not account:
            # Generate a random 4-digit account number
            new_acc_num = str(random.randint(1000, 9999))
            
            # Ensure it's unique (simple while loop just in case)
            while db.query(Account).filter(Account.account_number == new_acc_num).first():
                new_acc_num = str(random.randint(1000, 9999))
                
            new_account = Account(
                account_number=new_acc_num,
                user_id=user.user_id,
                account_type="checking",
                balance=0.0
            )
            db.add(new_account)
            db.commit()

        # Issue the JWT containing the user_id as the subject ("sub")
        access_token = create_access_token(data={"sub": str(user.user_id)})
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user_id": user.user_id
        }