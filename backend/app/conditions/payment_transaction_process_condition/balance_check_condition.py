from app.schemas.payment_transaction_process_schema import TransferState

def balance_check_condition(state: TransferState):
    
    # 1. Database level failure (e.g., account does not exist)
    if state['balance_check_status'] == 'failed':
        return 'rejected' 

    # 2. Logic level failure (e.g., not enough money)
    if state['balance_check_status'] == 'insufficient_balance':
        return 'notify_customer'
    
    # 3. Success - proceed to security layer
    if state['balance_check_status'] == 'passed':
        return 'fraud_check'
    
    # Safety fallback to prevent infinite loops if status is missing
    return 'rejected'