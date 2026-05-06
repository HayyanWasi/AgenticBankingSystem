
from app.schemas.kyc_agent_schema import UserContext
from app.agents.kyc_agents.mockdata import MOCK_DB
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from datetime import datetime



@tool("verify_customer_id", 
      description="verify the customer id from the database. if not found, dismiss verification",
      args_schema=UserContext)
def verify_id(id_num: str, full_name: str) -> dict:
    """Search customer in DB, verify ID and check expiry in one call"""
    customer = MOCK_DB.get(id_num)
    
    if not customer:
        return {"score": 0.0, "reason": "Customer not found"}
    
    name_match = customer["full_name"].lower() == full_name.lower()
    # Fixed: use datetime.strptime since we imported the class
    not_expired = datetime.strptime(customer["expiry_date"], "%Y-%m-%d") > datetime.now()
    
    score = 0.0
    if name_match: score += 0.4
    if not_expired: score += 0.4
    score += 0.2  # found in DB
    
    return {"score": score, "name_match": name_match, "not_expired": not_expired}

