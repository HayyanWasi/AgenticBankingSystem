from app.schemas.payment_transaction_process_schema import TransferState

def notify_customer(state: TransferState):
    
    if state.get('notification_message'):
        return {'notification_message': state['notification_message']}

    status = state.get('transaction_status', 'unknown')
    amount = state.get('money', '0.0')
    receiver = state.get('receiver_account_number', 'an unknown account')
    reason = state.get('reason', 'System error')

    if status == 'completed':
        message = f"Success! Your transfer of ${amount} to {receiver} has been completed."
    
    elif status == 'failed':
        message = f"Transfer Failed. We could not process your transfer of ${amount}. Reason: {reason}."
    
    else:
        message = f"Notice: Your transfer status is currently: {status}."

    return {
        'notification_message': message
    }