import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.graphs.kyc_agent_graph import kyc_app

initial_state = {
    "id_card_num": "112233445566",
    "full_name": "hayyan1",
    "phone_number": "0303-0303030",
    "nationality": "Pakistan",
    "kyc_score": 0.0,
    "messages": []
}

config = {"configurable": {"thread_id": "test_hang_112233445566"}}

print("Invoking graph...")
try:
    for event in kyc_app.stream(initial_state, config=config):
        for k, v in event.items():
            print(f"--> Finished Node: {k}")
except Exception as e:
    print(f"Exception: {e}")
print("Finished stream!")
