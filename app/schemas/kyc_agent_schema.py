from pydantic import BaseModel
from typing import TypedDict, Literal, Annotated
from langgraph.graph import add_messages

class KYCState(TypedDict):
    # customer inputs
    id_card_num: str
    phone_number: str
    full_name: str
    issue_date: str
    expiry_date: str
    nationality: str
    date_of_birth: str
    place_of_birth: str
    gender: str
    
    # result fields
    kyc_score: float
    verification_status: str
    human_decision: str
    rejection_reason: str
    notification_message: str
    kyc_status: str

    messages: Annotated[list, add_messages]

class VerificationResult(BaseModel):
    kyc_score: float
    status: Literal['approved', 'rejected', 'human_review']
    rejection_reason: str


    
    