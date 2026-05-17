from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True)
    full_name = Column(String, nullable=False)
    id_card_num = Column(String, unique=True, nullable=False)
    phone_number = Column(String)
    nationality = Column(String)
    balance = Column(Float, default=0.0)

    # Relationships
    accounts = relationship("Account", back_populates="owner")
    loans = relationship("Loan", back_populates="borrower")
    kyc_record = relationship("KYC", back_populates="user", uselist=False)

class Account(Base):
    __tablename__ = 'accounts'

    account_id = Column(Integer, primary_key=True)
    account_number = Column(String, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    account_type = Column(String, nullable=False)
    balance = Column(Float, default=0.0)
    status = Column(String, default="active")

    # Relationships
    owner = relationship("User", back_populates="accounts")
    # Link transactions sent from this account
    sent_transactions = relationship("Transaction", foreign_keys='Transaction.from_account_id', back_populates="sender")
    # Link transactions received by this account
    received_transactions = relationship("Transaction", foreign_keys='Transaction.to_account_id', back_populates="receiver")

# In your models.py
class Loan(Base):
    __tablename__ = 'loans'
    loan_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    
    # Inputs
    full_name = Column(String)
    loan_amount = Column(Float)
    loan_purpose = Column(String)
    monthly_income = Column(Float)
    loan_term_months = Column(Integer)
    
    # Logic Results (from your nodes)
    credit_score = Column(Integer)
    verified_income_ratio = Column(Float)
    underwriting_decision = Column(String) # auto_approve, human_review, etc.
    loan_status = Column(String)           # approved, rejected
    notification_message = Column(Text)
    
    # Relationship
    borrower = relationship("User", back_populates="loans")
class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(Integer, primary_key=True, index=True)
    from_account_id = Column(Integer, ForeignKey('accounts.account_id'), nullable=False)
    to_account_id = Column(Integer, ForeignKey('accounts.account_id'), nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String) 
    

    # Relationships
    sender = relationship("Account", foreign_keys=[from_account_id], back_populates="sent_transactions")
    receiver = relationship("Account", foreign_keys=[to_account_id], back_populates="received_transactions")

class KYC(Base):
    __tablename__ = "kyc"

    kyc_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), unique=True, nullable=False)
    kyc_score = Column(Float)
    verification_status = Column(String, default="pending")
    reject_reason = Column(String)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)
    audit_trail = Column(Text, nullable=True)


    user = relationship("User", back_populates="kyc_record")