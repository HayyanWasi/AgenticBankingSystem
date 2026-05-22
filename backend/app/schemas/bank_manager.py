from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]

    next_route: str

    # ── Injected user context (pre-loaded from JWT + DB, never asked from user) ──
    user_id: Optional[int]
    full_name: Optional[str]
    id_card_num: Optional[str]
    sender_account_number: Optional[str]  # user's own primary account number

    # Status flags so the Supervisor knows where the user is in the journey
    kyc_status: Optional[str]
    loan_status: Optional[str]
    transfer_status: Optional[str]
