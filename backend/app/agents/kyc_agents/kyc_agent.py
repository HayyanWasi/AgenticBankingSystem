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
                You are a Senior Bank Compliance Officer. Assess the risk for:
                Name: {state['full_name']}
                ID: {state['id_card_num']}

                ### PROTOCOL:
                1. Call 'verify_customer_id' with id_num='{state['id_card_num']}' and full_name='{state['full_name']}'.
                2. The tool returns a score (0.0-1.0) and a status. Report your findings clearly.
                3. Do NOT call any other tools. Only 'verify_customer_id' is available."""
                            ),
            HumanMessage(
                content="Please verify this customer now."
            )
        ]
        response = llm_with_tools.invoke(initial_messages)
        return {"messages": initial_messages + [response]}
    else:
        # Subsequent calls: full history already in state
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}