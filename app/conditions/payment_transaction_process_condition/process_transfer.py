from app.schemas.payment_transaction_process_schema import TransferState

def process_transfer(state: TransferState):
    return {
        'transaction_status': 'approved',
        'notification_message': f"Dear {state['user']}, your transfer of {state['money']} has been approved to account number {state['to_transfer_acc_num']}."
    }