from app.agents.kyc_agents.notify_customer import notify_customer
from app.agents.kyc_agents.human_review_kyc import human_review
from app.agents.kyc_agents.kyc_agent import kyc_agent
from app.agents.kyc_agents.kyc_agent_results import kyc_agent_results
from app.schemas.kyc_agent_schema import KYCState
from app.conditions.kyc_process_conditions.kyc_tool_conditions import tools_condition
from app.conditions.kyc_process_conditions.score_conditions import score_condition
from app.agents.kyc_agents.human_review_kyc import human_review
from app.agents.kyc_agents.kyc_agent import kyc_agent
from app.agents.kyc_agents.kyc_agent_results import kyc_agent_results
from app.schemas.kyc_agent_schema import KYCState
from app.conditions.kyc_process_conditions.kyc_tool_conditions import tools_condition
from app.conditions.kyc_process_conditions.score_conditions import score_condition
from app.config.kyc_config import tool_node 

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

graph = StateGraph(KYCState)

graph.add_node("kyc_agent", kyc_agent)
graph.add_node("action", tool_node)
graph.add_node("kyc_agent_results", kyc_agent_results)
graph.add_node('human_review', human_review)
graph.add_node('notify_customer', notify_customer)


graph.add_edge(START, "kyc_agent")
graph.add_conditional_edges(
    "kyc_agent",
    tools_condition,
    {
        "action": "action",
        "results": "kyc_agent_results"
    }
)
graph.add_edge("action", "kyc_agent")
graph.add_conditional_edges(
    "kyc_agent_results",
    score_condition,
    {
        "approve": "notify_customer",
        "human_review": "human_review",
        "reject": "notify_customer"
    }
)
graph.add_edge("human_review", "notify_customer")
graph.add_edge("notify_customer", END)


checkPointer = MemorySaver()

workflow = graph.compile(checkpointer=checkPointer)

#TEST CASE 1: valid inputs

# initial_state = {
#     "id_card_num": "42101-1234567-1",
#     "full_name": "John Doe",
#     "phone_number": "0300-1234567",
#     "issue_date": "2020-01-01",
#     "expiry_date": "2028-01-01",
#     "nationality": "Pakistani",
#     "date_of_birth": "1990-01-01",
#     "place_of_birth": "Karachi",
#     "gender": "Male",
#     "kyc_score": 0.0,
#     "verification_status": "",
#     "human_decision": "",
#     "rejection_reason": "",
#     "notification_message": "",
#     "kyc_status": "",
#     "messages": []
# }

# config = {"configurable": {"thread_id": "kyc-test-1"}}
# result = workflow.invoke(initial_state, config=config)
# print(result)




#TEST CASE 2: human review 

# initial_state = {
#     "id_card_num": "42201-5544332-7",  # Fatima Ali
#     "full_name": "Fatima Ali",
#     "phone_number": "0300-1234567",
#     "issue_date": "2020-01-01",
#     "expiry_date": "2028-01-01",
#     "nationality": "Pakistani",
#     "date_of_birth": "1990-01-01",
#     "place_of_birth": "Karachi",
#     "gender": "Male",
#     "kyc_score": 0.0,
#     "verification_status": "",
#     "human_decision": "",
#     "rejection_reason": "",
#     "notification_message": "",
#     "kyc_status": "",
#     "messages": []
# }
# config2 = {"configurable": {"thread_id": "kyc-test-2b"}}
# result = workflow.invoke(initial_state, config=config2)
# result = workflow.invoke(Command(resume="reject"), config=config2)
# print("Score:", result['kyc_score'])
# print("Status:", result['verification_status'])
# print("Notification:", result['notification_message'])
# print("Human Decision:", result['human_decision'])
# print("REJECTED:", result['notification_message'])



#TEST CASE 3: 
initial_state = {
   "id_card_num": "00000-0000000-0",
    "full_name": "Nobody",
    "phone_number": "0300-1234567",
    "issue_date": "2020-01-01",
    "expiry_date": "2028-01-01",
    "nationality": "Pakistani",
    "date_of_birth": "1990-01-01",
    "place_of_birth": "Karachi",
    "gender": "Male",
    "kyc_score": 0.0,
    "verification_status": "",
    "human_decision": "",
    "rejection_reason": "",
    "notification_message": "",
    "kyc_status": "",
    "messages": []
}
# Scenario 3 - correct test
config3 = {"configurable": {"thread_id": "kyc-test-3"}}
result = workflow.invoke(initial_state, config=config3)
print("Score:", result['kyc_score'])
print("Status:", result['verification_status'])
print("Notification:", result['notification_message'])
print("Status:", result['verification_status'])
print("Notification:", result['notification_message'])