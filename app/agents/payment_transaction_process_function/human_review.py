from langgraph.types import interrupt

from app.schemas.payment_transaction_process_schema import TransferState

def human_review(state: TransferState):
    decision = interrupt({
        "user": state['user'],
        "amount": state['money'],
        "recipient": state['to_transfer_acc_num'],
        "fraud_score": state['fraud_score'],
        "time": state['sent_time'],
        "question": "Approve or reject this transaction?"
    })
    return {'human_decision': decision}
    
   