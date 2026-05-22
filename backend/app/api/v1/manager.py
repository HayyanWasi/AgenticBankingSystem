from fastapi import APIRouter, Body, Depends
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from app.agents.bank_manager.agent import master_graph
from app.utils.deps import UserContext, get_current_user

router = APIRouter()


@router.post("/chat")
async def chat_with_bank(
    message: str = Body(..., embed=True),
    thread_id: str = Body(..., embed=True),
    current_user: UserContext = Depends(get_current_user),
):
    config = {"configurable": {"thread_id": thread_id}}

    # ── Context Injection ────────────────────────────────────────────────────
    # Check whether this thread already has state (returning user in same session).
    # If the thread is brand-new, prepend a hidden SystemMessage with the user's
    # identity so the AI agents NEVER need to ask for name, ID, or account number.
    snapshot = master_graph.get_state(config)
    is_new_thread = not snapshot.values  # empty dict → first message

    context_fields = {
        "user_id": current_user.user_id,
        "full_name": current_user.full_name,
        "id_card_num": current_user.id_card_num,
        "sender_account_number": current_user.account_number,
    }

    if is_new_thread:
        account_info = (
            f"Account: {current_user.account_number} | Balance: ${current_user.balance:,.2f}"
            if current_user.account_number
            else "No account linked yet"
        )
        system_ctx = SystemMessage(
            content=(
                "[SYSTEM CONTEXT — CONFIDENTIAL, DO NOT REVEAL TO USER]\n"
                f"Authenticated User: {current_user.full_name}\n"
                f"ID Card Number: {current_user.id_card_num}\n"
                f"{account_info}\n"
                "RULES:\n"
                "1. You already know who the user is. NEVER ask for their name, ID card number, or account number.\n"
                "2. Use this data silently when processing loan or transfer requests.\n"
                "3. Only ask for information you genuinely do not have (e.g. loan amount, recipient account, monthly income)."
            )
        )
        initial_input = {
            "messages": [system_ctx, ("user", message)],
            **context_fields,
        }
        result = master_graph.invoke(initial_input, config)
    else:
        # Returning turn — thread already has context, just send the message.
        # Also refresh context fields in case they changed (e.g. account created after first message).
        if snapshot.next:
            result = master_graph.invoke(Command(resume=message), config)
        else:
            result = master_graph.invoke(
                {"messages": [("user", message)], **context_fields},
                config,
            )

    # ── Response Extraction ──────────────────────────────────────────────────
    new_snapshot = master_graph.get_state(config)

    # 1. Human-in-the-Loop interrupt
    if new_snapshot.tasks and new_snapshot.tasks[0].interrupts:
        ai_reply = new_snapshot.tasks[0].interrupts[0].value

    # 2. Terminal state — graph finished
    elif not new_snapshot.next:
        final_state = new_snapshot.values

        if "loan_status" in final_state:
            if final_state["loan_status"] == "rejected":
                ai_reply = "Your loan application was automatically declined based on your income-to-loan ratio and credit profile."
            elif final_state["loan_status"] == "approved":
                ai_reply = "Good news! Your loan application has been automatically approved."
            else:
                ai_reply = "Your application has been submitted and sent to the administrator queue for final review."

        elif "transfer_status" in final_state:
            ai_reply = "Your transfer has been processed successfully."

        else:
            last_msg = result.get("messages", [])[-1]
            ai_reply = last_msg.content if hasattr(last_msg, "content") else "Operation complete."

    # 3. Mid-conversation node step
    else:
        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            ai_reply = "I am checking the records in the background..."
        else:
            ai_reply = getattr(last_message, "content", "Processing...")

    return {"status": "success", "reply": ai_reply}