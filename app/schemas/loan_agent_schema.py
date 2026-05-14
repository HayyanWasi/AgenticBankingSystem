from typing import TypedDict, List, Literal, Optional
import operator
from typing import Annotated
from langgraph.graph.message import add_messages


class LoanState(TypedDict):
    # chat history — passed in from the supervisor
    messages: Annotated[list, add_messages]

    # application inputs — Optional because they are populated by the extraction node
    full_name: Optional[str]
    id_card_num: Optional[str]
    loan_amount: Optional[float]
    loan_term_months: Optional[int]
    monthly_income: Optional[float]
    loan_purpose: Optional[Literal['personal', 'business', 'mortgage']]
    loan_reason: Optional[str]
    
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
    loan_id: Optional[int]
    
    # notification
    notification_message: str
    loan_status: str