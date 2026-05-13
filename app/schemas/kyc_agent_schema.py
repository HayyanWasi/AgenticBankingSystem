from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, Optional
from langgraph.graph import add_messages

class KYCState(TypedDict, total=False):
    # Customer inputs — Maps perfectly to the User table
    id_card_num: Optional[str]
    phone_number: Optional[str]
    full_name: Optional[str]
    nationality: Optional[str]
    
    # Result fields — Maps perfectly to the KYC table
    kyc_score: Optional[float]
    verification_status: Optional[str] 
    human_decision: Optional[str]
    reject_reason: Optional[str]       
    notification_message: Optional[str]
    
    # Graph execution history
    messages: Annotated[list, add_messages]

class VerificationResult(BaseModel):
    kyc_score: float = Field(ge=0.0, le=1.0, description="KYC score between 0 and 1")
    status: Literal['approved', 'rejected', 'human_review']
    reject_reason: str

class UserContext(BaseModel):
    id_num: str = Field(description="Customer ID card number")
    full_name: str = Field(description="Customer full name")