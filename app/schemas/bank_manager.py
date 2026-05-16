from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]

    next_route: str

    # Shared context needed by all sub-agents
    user_id: Optional[int]
    id_card_num: Optional[str]
    
    # Status flags so the Supervisor knows where the user is in the journey
    kyc_status: Optional[str]
    loan_status: Optional[str]
    transfer_status: Optional[str]
