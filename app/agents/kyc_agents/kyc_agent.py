from app.config.kyc_config import llm_with_tools
from app.schemas.kyc_agent_schema import KYCState
from langchain_core.messages import SystemMessage, HumanMessage


def kyc_agent(state: KYCState):
    """
    KYC agent node — calls LLM with tools to verify the customer.
    """
    messages = state['messages']
    if not messages:
        # First call: build initial messages
        initial_messages = [
            SystemMessage(
                content=f"""
                You are a KYC agent. Verify customer ID:
                {state['id_card_num']}, 
                Name: {state['full_name']}. 
                Score < 0.4 = reject, 
                0.4-0.7 = human_review,
                > 0.7 = approve."""
            ),
            HumanMessage(
                content="verify this customer using your tools."
            )
        ]
        response = llm_with_tools.invoke(initial_messages)
        return {"messages": initial_messages + [response]}
    else:
        # Subsequent calls: full history already in state
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}