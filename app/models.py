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

class Loan(Base):
    __tablename__ = 'loans' 

    loan_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    amount = Column(Float, nullable=False)
    term_months = Column(Integer, nullable=False)
    interest_rate = Column(Float, nullable=False)
    status = Column(String, default="pending")
    loan_purpose = Column(String)
    monthly_income = Column(Float)

    # Relationships
    borrower = relationship("User", back_populates="loans")

class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(Integer, primary_key=True)
    from_account_id = Column(Integer, ForeignKey('accounts.account_id'), nullable=False)
    to_account_id = Column(Integer, ForeignKey('accounts.account_id'), nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

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