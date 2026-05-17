from app.schemas.kyc_agent_schema import KYCState

def tools_condition(state: KYCState):
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM wants to use a tool, go to the "action" node
    #Human ne query di example user ki id nikal k lao -> AI ne bola i dont know the answer i have specific tool 
    #to verify it Ai calls tool using tool_calls is s technically an attribute (a piece of data) of the AIMessage object
    # action is attached with this node graph.add_node("action", tool_node) graph is calling the tool_node


    if last_message.tool_calls:
        return "action"
    #if we dont need tool we just simply hit the results END node 
    # Otherwise, we are done with tools, go to extract final results
    return "results" 