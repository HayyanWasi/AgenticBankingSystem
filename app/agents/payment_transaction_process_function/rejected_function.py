from app.schemas.payment_transaction_process_schema import TransferState

def rejected(state: TransferState):
    return {
        'transaction_status': 'rejected',
        'notification_message': f"Dear {state['user']}, your transfer of {state['money']} was rejected after manual review."
    }