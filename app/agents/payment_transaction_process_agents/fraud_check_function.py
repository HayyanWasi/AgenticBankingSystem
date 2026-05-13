from datetime import datetime
from schemas.payment_transaction_process_schema import TransferState

def fraud_check(state: TransferState):
    # 1. Safely extract values to prevent KeyErrors and handle explicit None values
    money = state.get('money') or 0.0
    balance = state.get('total_balance') or 0.0
    time_string = state.get('sent_time') or '12:00' # Default to safe hour if missing or None

    # 2. Prevent Division by Zero
    if balance <= 0:
        percent_used = 1.0 # Treat as 100% to trigger highest risk
    else:
        percent_used = money / balance
        
    # 3. Calculate Signal 1 (Amount vs Balance)
    signal_1 = 0.0
    if percent_used > 0.90:
        signal_1 = 0.9
    elif percent_used > 0.75:
        signal_1 = 0.7
    elif percent_used > 0.50:
        signal_1 = 0.5
    elif percent_used > 0.30:
        signal_1 = 0.2

    # 4. Calculate Signal 2 (Time of Day)
    signal_2 = 0.0
    try:
        # Assuming time_string is formatted like "HH:MM"
        hour = int(time_string.split(":")[0])
        if 0 <= hour < 5:
            signal_2 = 0.9
        elif hour == 22 or hour == 23 or hour == 5:
            signal_2 = 0.3
    except (ValueError, AttributeError):
        # If time parsing fails, assume moderate risk
        signal_2 = 0.5 

    # 5. Calculate Signal 3 (Recipient)
    signal_3 = 0.1 # Hardcoded per your rules
    
    # 6. Final Calculation
    fraud_score = (signal_1 * 0.4) + (signal_2 * 0.4) + (signal_3 * 0.2)
    is_fraud = bool(fraud_score > 0.7)
    
    return {
        'fraud_score': round(fraud_score, 2), 
        'is_fraud': is_fraud
    }