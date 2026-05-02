
from app.schemas.payment_transaction_process_schema import TransferState

def notify_customer(state: TransferState):
    
    if state['transaction_status'] == 'approved':
        message = f"Dear {state['user']}, your transfer of {state['money']} has been approved to account number {state['to_transfer_acc_num']}."
    elif state['transaction_status'] == 'failed':
        message = f"Dear {state['user']}, your transfer of {state['money']} failed. Your current balance is {state['total_balance']}."
    elif state['transaction_status'] == 'rejected':
        message = f"Dear {state['user']}, your transfer of {state['money']} was rejected after manual review."
    return {
        'notification_message': message,
    }