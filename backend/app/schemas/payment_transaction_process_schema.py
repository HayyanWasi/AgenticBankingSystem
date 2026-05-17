from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class TransferState(TypedDict, total=False):
    # chat history
    messages: Annotated[list, add_messages]

    # user inputs — Populated by extraction
    user: Optional[str]
    sender_account_number: Optional[str]   # Fixed naming & type
    receiver_account_number: Optional[str] # Fixed naming & type
    money: Optional[float]                 # Fixed type
    sent_time: Optional[str]
    total_balance: Optional[float]         # Fixed type

    # flow results 
    balance_check_status: Optional[str]
    reason: Optional[str]                  # Added for DB error handling
    fraud_score: Optional[float]
    is_fraud: Optional[bool]
    human_decision: Optional[str]
    transaction_status: Optional[str]
    notification_message: Optional[str]

# Unrelated to state, but good to keep if you use it elsewhere
class FraudCheck(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0, description="Fraud score between 0 and 1")
    is_fraud: bool