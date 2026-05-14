from schemas.loan_agent_schema import LoanState

def income_verification(state: LoanState)->dict:
    loan_amount = state['loan_amount']
    monthly_income = state['monthly_income']
    loan_term_months = state["loan_term_months"]

    verified_income_ratio = (loan_amount / loan_term_months) / monthly_income
    
    return {"verified_income_ratio": verified_income_ratio}
