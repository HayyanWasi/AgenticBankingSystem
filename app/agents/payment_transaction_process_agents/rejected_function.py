from schemas.payment_transaction_process_schema import TransferState

def rejected(state: TransferState):
    # Fetch the specific reason, default to "manual review" if none is provided
    failure_reason = state.get('reason', 'failed manual review')
    
    return {
        'transaction_status': 'rejected',
        'notification_message': f"Dear {state.get('user', 'Customer')}, your transfer of {state.get('money')} was rejected. Reason: {failure_reason}."
    }