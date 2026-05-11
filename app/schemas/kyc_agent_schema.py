from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, Optional
from langgraph.graph import add_messages

class KYCState(TypedDict):
    # customer inputs — Optional because they are populated by the extraction node
    id_card_num: Optional[str]
    phone_number: Optional[str]
    full_name: Optional[str]
    issue_date: Optional[str]
    expiry_date: Optional[str]
    nationality: Optional[str]
    date_of_birth: Optional[str]
    place_of_birth: Optional[str]
    gender: Optional[str]
    
    # result fields
    kyc_score: float
    verification_status: Literal['approved', 'rejected', 'human_review']
    human_decision: str
    rejection_reason: str
    notification_message: str
    kyc_status: str

    messages: Annotated[list, add_messages]

class VerificationResult(BaseModel):
    kyc_score: float = Field(ge=0.0, le=1.0, description="KYC score between 0 and 1" )
    status: Literal['approved', 'rejected', 'human_review']
    rejection_reason: str


    
class UserContext(BaseModel):
    id_num: str = Field(description="Customer ID number")
    full_name: str = Field(description="Customer Full Name")