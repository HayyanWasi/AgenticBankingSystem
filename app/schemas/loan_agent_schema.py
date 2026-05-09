from typing import TypedDict, List, Literal, Optional
import operator
from typing import Annotated


class LoanState(TypedDict):
    # application inputs
    full_name: str
    id_card_num: str
    loan_amount: float
    loan_term_months: int
    monthly_income: float
    loan_purpose: Literal['personal', 'business', 'mortgage']
    loan_reason: str
    
    # verification results
    credit_score: int
    credit_flags: Annotated[List[str], operator.add]
    verified_income_ratio: float
    
    # risk results
    risk_tier: Literal['low', 'medium', 'high']
    
    # decision results
    underwriting_decision: Literal['auto_approve', 'auto_decline', 'human_review']
    human_review_result: Optional[Literal['approved', 'denied']]
    
    # workflow
    tasks: List[str]
    workflow_status: Literal['pending', 'running', 'completed', 'failed']
    
    # notification
    notification_message: str
    loan_status: str