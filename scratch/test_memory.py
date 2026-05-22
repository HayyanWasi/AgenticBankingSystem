import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from app.graphs.loan_agent_graph import memory

print(dir(memory))
if hasattr(memory, 'storage'):
    print("Storage keys:", memory.storage.keys())
