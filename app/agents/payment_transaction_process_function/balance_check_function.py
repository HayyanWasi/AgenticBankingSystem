from app.schemas.payment_transaction_process_schema import TransferState

def balance_check(state: TransferState):
    
    if state['money'] <= state['total_balance']:
        return {"balance_check_status": 'passed'}
    return {
        "balance_check_status": 'insufficient_balance',
        "transaction_status": 'failed'
    }