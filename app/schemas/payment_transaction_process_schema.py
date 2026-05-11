from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class TransferState(TypedDict):
    # chat history — passed in from the supervisor
    messages: Annotated[list, add_messages]

    # user inputs — Optional because they are populated by the extraction node
    user: Optional[str]
    account_num: Optional[int]
    to_transfer_acc_num: Optional[int]
    money: Optional[int]
    sent_time: Optional[str]
    total_balance: Optional[int]

    # flow results — populated during processing
    balance_check_status: str
    fraud_score: float
    is_fraud: bool
    human_decision: str
    transaction_status: str
    notification_message: str

class FruadCheck(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0, description="Fraud score between 0 and 1")
    is_fraud: bool
