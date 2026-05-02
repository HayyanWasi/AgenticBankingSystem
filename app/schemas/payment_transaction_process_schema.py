from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field 



class TransferState(TypedDict):
    # user inputs
    user: str
    account_num: int
    to_transfer_acc_num: int
    money: int
    sent_time: str
    
    # flow results
    balance_check_status: str
    fraud_score: float
    is_fraud: bool
    human_decision: str
    transaction_status: str
    notification_message: str
    total_balance: int

class FruadCheck(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0, description="Fraud score between 0 and 1")
    is_fraud: bool
