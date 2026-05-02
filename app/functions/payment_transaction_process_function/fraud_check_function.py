from langchain_core.messages import HumanMessage, SystemMessage
from app.schemas.payment_transaction_process_schema import TransferState
from app.config.config import structured_evaluator_llm

## need to add hugging face model for fraud check when adding this code into seperate file
def fraud_check(state: TransferState):
    
    messages=[
        SystemMessage(
            content=f"""
            You are a **Fruad Detection agent**. you're goal is to analyze the if the transaction is fraud or not.
            Things to judge the whether tranaction is fraud or not:
            1. Is the transfer amount unusually large? (e.g. more than 50% of total balance): total balance: {state['total_balance']} 
            2. Is it an unusual hour? (e.g. 2am - 5am transactions are suspicious)
            3. Is the recipient new or unknown?

            Transaction details:
            Amount: {state['money']}
            Time: {state['sent_time']}
            Recipient: {state['to_transfer_acc_num']}
            Balance Check Status: {state['balance_check_status']}

            """,
        ),
        HumanMessage(content="Analyze this transaction and return your fraud assessment.")
    ]
    response=structured_evaluator_llm.invoke(messages)
    return {'fraud_score': response.fraud_score, 'is_fraud': response.is_fraud}
