import datetime
import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from app.schemas.payment_transaction_process_schema import TransferState, FruadCheck
from app.agents.payment_transaction_process_function.balance_check_function import balance_check
from app.agents.payment_transaction_process_function.fraud_check_function import fraud_check
from app.agents.payment_transaction_process_function.notify_customer_functions import notify_customer
from app.agents.payment_transaction_process_function.human_review import human_review
from app.agents.payment_transaction_process_function.rejected_function import rejected
from app.conditions.payment_transaction_process_condition.process_transfer import process_transfer
from app.conditions.payment_transaction_process_condition.balance_check_condition import balance_check_condition
from app.conditions.payment_transaction_process_condition.fraud_check_condition import fraud_check_condition
from app.conditions.payment_transaction_process_condition.human_review_condition import human_review_condition

load_dotenv()

transfer_workflow = StateGraph(TransferState)

#nodes
transfer_workflow.add_node("balance_check", balance_check)
transfer_workflow.add_node("fraud_check", fraud_check)
transfer_workflow.add_node("notify_customer", notify_customer)
transfer_workflow.add_node("human_review", human_review)
transfer_workflow.add_node("process_transfer", process_transfer)
transfer_workflow.add_node("rejected", rejected)

#edges
transfer_workflow.add_edge(START, "balance_check")
transfer_workflow.add_conditional_edges('balance_check', balance_check_condition)
transfer_workflow.add_conditional_edges('fraud_check', fraud_check_condition)
transfer_workflow.add_conditional_edges('human_review', human_review_condition)
transfer_workflow.add_edge('process_transfer', 'notify_customer')
transfer_workflow.add_edge('rejected', 'notify_customer')
transfer_workflow.add_edge('notify_customer', END)

transfer_workflow=transfer_workflow.compile(checkpointer=MemorySaver())
print(transfer_workflow.get_graph().draw_mermaid())



# #Test code
# initial_state = {
#     "user": "John Doe",
#     "account_num": 123456,
#     "to_transfer_acc_num": 999999,
#     "money": 15000,
#     "total_balance": 10000,
#     "sent_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "balance_check_status": "",
#     "fraud_score": 0.4,
#     "is_fraud": True,
#     "human_decision": "",
#     "transaction_status": "",
#     "notification_message": ""
# }

# config = {"configurable": {"thread_id": "test-3"}}

# result = transfer_workflow.invoke(initial_state, config=config)
# print(result)

# # # to approve
# # result = transfer_workflow.invoke(
# #     Command(resume="approve"),
# #     config=config
# # )
# # print(result)

# # to reject
# result = transfer_workflow.invoke(
#     Command(resume="reject"),
#     config=config
# )
# print(result)

# test reject scenario
initial_state = {
    "user": "John Doe",
    "account_num": 123456,
    "to_transfer_acc_num": 999999,
"money": 9500,
"total_balance": 10000,  # 95% of balance
"sent_time": "2026-05-03 03:00:00",  # 3am  # normal hour # normal hour
    "fraud_score": 0.0,
    "is_fraud": False,
    "human_decision": "",
    "transaction_status": "",
    "notification_message": ""
}

config = {"configurable": {"thread_id": "test-reject-1"}}

# first invoke - will interrupt at human review
result = transfer_workflow.invoke(initial_state, config=config)
print("INTERRUPTED:", result)

# resume with reject
result = transfer_workflow.invoke(Command(resume="reject"), config=config)
print("REJECTED:", result)
#add business logic in notify