from langchain_core.messages import HumanMessage
from app.schemas.bank_manager import SupervisorState

from langgraph.graph.state import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graphs.kyc_agent_graph import kyc_app as kyc_graph
from app.graphs.loan_agent_graph import loan_app as loan_agent_graph
from app.graphs.transfer_graph import transfer_workflow as transfer_agent_graph
from app.graphs.privacy_agent_graph import privacy_policy_node
from app.config.supervisor_config import route_decision_llm


def supervisor_route(state: SupervisorState) -> dict:
    last_message = state["messages"][-1].content
    
    prompt = f"""
    Act as a bank dispatcher. Route this user request: "{last_message}"
    
    - If they ask about policies, sharing, or privacy: privacy_policy_agent
    - If they ask for a loan: loan_agent
    - If they want to send money: payment_transaction_process_agent
    - If they mention ID/Verification: kyc_agent
    - If it's a greeting or closing: END
    """
    
    response = route_decision_llm.invoke(prompt)
    
    destination = response.destination 

    print(f"--- [ROUTING] Decision: {destination} ---")
    return {"next_route": destination}  
    
graph = StateGraph(SupervisorState)

graph.add_node("supervisor", supervisor_route)
# Each sub-graph handles its own data extraction as its first node
graph.add_node("loan_agent", loan_agent_graph)
graph.add_node("privacy_policy_agent", privacy_policy_node)
graph.add_node("payment_transaction_process_agent", transfer_agent_graph)
graph.add_node("kyc_agent", kyc_graph)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", lambda state: state["next_route"])
graph.add_edge("loan_agent", END)
graph.add_edge("payment_transaction_process_agent", END)
graph.add_edge("kyc_agent", END)
graph.add_edge("privacy_policy_agent", END)

master_graph = graph.compile(checkpointer=MemorySaver())


# ── CLI Test Runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=== AGENTIC BANKING SYSTEM ===")
    user_input = input("You: ").strip()

    # Supervisor only needs the user's message — sub-graphs handle extraction themselves
    initial_state = {"messages": [("user", user_input)]}

    print("\n--- Routing request ---")

    interrupted = False
    for event in master_graph.stream(initial_state, config, stream_mode="updates"):
        node = list(event.keys())[0]
        data = event[node]

        if node == "__interrupt__":
            # data is a tuple of Interrupt objects; .value is the question text
            question = data[0].value
            print(f"\nAgent: {question}")
            interrupted = True
            break
        else:
            print(f"[{node}] routed")

    # Conversation loop: keep responding until the graph finishes
    while interrupted:
        user_reply = input("You: ").strip()
        interrupted = False

        for event in master_graph.stream(
            Command(resume=user_reply), config, stream_mode="updates"
        ):
            node = list(event.keys())[0]
            data = event[node]

            if node == "__interrupt__":
                question = data[0].value
                print(f"\nAgent: {question}")
                interrupted = True
                break
            else:
                # Print any final messages returned by sub-graph nodes
                if isinstance(data, dict):
                    for msg in data.get("messages", []):
                        if hasattr(msg, "content") and msg.content:
                            print(f"\nAgent: {msg.content}")

    print("\n=== SESSION COMPLETE ===")