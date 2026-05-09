from app.schemas.loan_agent_schema import LoanState



def credit_verification(state: LoanState) -> dict:
    monthly_income = state['monthly_income']
    loan_amount = state['loan_amount']
    if monthly_income > 50000:
        score = 750
    elif monthly_income >= 30000:
        score = 650
    else:
        score = 550


    flags = []
    if monthly_income < 30000:
        flags.append("low_income")
    if loan_amount > (10 * monthly_income):
        flags.append("high_debt_ratio")

    return {
        "credit_score": score,
        "credit_flags": flags
    }
