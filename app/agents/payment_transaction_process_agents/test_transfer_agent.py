from langchain_core.messages import HumanMessage
from langgraph.types import Command
# Adjust this import to point to where you actually compile your transfer_workflow
from app.graphs.transfer_graph import transfer_workflow

def run_test():
    print("--- Starting Transfer Agent Pipeline Test ---")
    
    # 1. Define a thread ID so LangGraph can save the state in memory
    config = {"configurable": {"thread_id": "test_thread_001"}}
    
    initial_input = "Hi, I need to send 150 dollars."
    print(f"\nUser: {initial_input}")
    
    # We use Command to send the initial input
    state_update = {"messages": [HumanMessage(content=initial_input)]}
    
    while True:
        # Run the graph until it hits an interrupt or finishes
        for event in transfer_workflow.stream(state_update, config, stream_mode="updates"):
            for node_name, node_data in event.items():
                print(f"\n[Node Executed: {node_name}]")
                if "balance_check_status" in node_data:
                    print(f"  Status: {node_data.get('balance_check_status')}")
                if "transaction_status" in node_data:
                    print(f"  Transaction: {node_data.get('transaction_status')}")
        
        # 4. Check the current state of the graph
        current_state = transfer_workflow.get_state(config)
        
        # If the graph has no next nodes, it has reached the END
        if not current_state.next:
            print("\n--- Pipeline Execution Complete ---")
            break
            
        # If the graph has next nodes but stopped, it hit our interrupt()
        tasks = current_state.tasks
        if tasks and tasks[0].interrupts:
            # Extract the question the LLM generated
            teller_question = tasks[0].interrupts[0].value
            
            print(f"\nTeller: {teller_question}")
            user_reply = input("Your Reply: ")
            
            # Resume the graph by passing the reply directly to the interrupt
            state_update = Command(resume=user_reply)
        else:
            break

if __name__ == "__main__":
    run_test()