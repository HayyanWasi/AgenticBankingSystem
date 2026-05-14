from langchain.tools import tool
from schemas.kyc_agent_schema import UserContext
from database.db_tool import get_user_kyc_status

HIGH_RISK_COUNTRIES = ["North Korea", "Iran", "Afghanistan", "Ukraine", "Syria"]

@tool("verify_customer_id",
      description="Look up a customer by their ID card number and assess their KYC risk. Returns a score and status.",
      args_schema=UserContext)
def verify_id(id_num: str, full_name: str) -> dict:
    """Query the real database for the customer and compute a risk score."""
    result = get_user_kyc_status(id_card_num=id_num)

    if result["status"] == "not_found":
        return {"score": 0.0, "reason": "Customer not found in database."}

    # Start from a base of 1.0 and apply deductions
    score = 1.0
    reasons = []

    # Check 1: High-risk nationality
    if result.get("is_high_risk"):
        score -= 0.4
        reasons.append(f"High-risk nationality: {result['nationality']}")

    # Check 2: Name mismatch
    db_name = (result.get("full_name") or "").lower().strip()
    provided_name = (full_name or "").lower().strip()
    if db_name and db_name != provided_name:
        score -= 0.3
        reasons.append(f"Name mismatch: DB has '{result['full_name']}', provided '{full_name}'")

    # Check 3: Incomplete profile (missing phone)
    if not result.get("phone_number"):
        score -= 0.2
        reasons.append("Incomplete profile: missing phone number")

    score = max(0.0, round(score, 2))

    if score >= 0.8:
        status = "approved"
    elif score >= 0.4:
        status = "human_review"
    else:
        status = "rejected"

    return {
        "score": score,
        "status": status,
        "nationality": result.get("nationality"),
        "is_high_risk": result.get("is_high_risk", False),
        "reason": "; ".join(reasons) if reasons else None,
    }
