from langchain_core.messages import HumanMessage, SystemMessage
from app.schemas.payment_transaction_process_schema import TransferState
from app.config.payment_transaction_process.config import structured_evaluator_llm

## need to add hugging face model for fraud check when adding this code into separate file
def fraud_check(state: TransferState):

    messages=[
        SystemMessage(
            content=f"""
            You are a **Fraud Detection agent**. Analyze the transaction below and return a fraud_score (0.0–1.0) and is_fraud (bool).

            ## Scoring Rules — evaluate each signal independently, then compute a WEIGHTED average:

            **Signal 1 — Transfer amount vs balance (weight: 40%)**
            - ≤ 30% of balance  → score 0.0  (normal)
            - 31–50% of balance → score 0.2  (slightly elevated)
            - 51–75% of balance → score 0.5  (moderate risk)
            - 76–90% of balance → score 0.7  (high risk)
            - > 90% of balance  → score 0.9  (very high risk)
            Total balance: {state['total_balance']} | Transfer amount: {state['money']}
            Percentage used: {round(state['money'] / state['total_balance'] * 100, 1)}%

            **Signal 2 — Transaction hour (24h format) (weight: 40%)**
            - 06:00–22:00 → score 0.0  (normal hours)
            - 22:00–00:00 or 05:00–06:00 → score 0.3  (late/early)
            - 00:00–05:00 → score 0.9  (highly suspicious hours)
            Transaction time: {state['sent_time']}

            **Signal 3 — Recipient account (weight: 20%)**
            - Since you cannot verify accounts, treat all as unknown/external → always score 0.1
            Recipient: {state['to_transfer_acc_num']}

            ## Final Score Calculation
            fraud_score = (Signal1 × 0.4) + (Signal2 × 0.4) + (Signal3 × 0.2)

            ## Final Decision
            - Set is_fraud = True ONLY if the computed fraud_score is ABOVE 0.7.
            - Set is_fraud = False if fraud_score is 0.7 or below.

            ## Calibration examples (follow these exactly):
            - 1000/10000 (10%) at 2pm  → S1=0.0, S2=0.0, S3=0.1 → (0.0×0.4)+(0.0×0.4)+(0.1×0.2) = 0.02  → is_fraud=False
            - 8000/10000 (80%) at 1am  → S1=0.7, S2=0.9, S3=0.1 → (0.7×0.4)+(0.9×0.4)+(0.1×0.2) = 0.66  → is_fraud=False
            - 9500/10000 (95%) at 3am  → S1=0.9, S2=0.9, S3=0.1 → (0.9×0.4)+(0.9×0.4)+(0.1×0.2) = 0.74  → is_fraud=True

            Balance Check Status: {state['balance_check_status']}
            """,
        ),
        HumanMessage(content="Analyze this transaction and return your fraud assessment.")
    ]
    response = structured_evaluator_llm.invoke(messages)
    return {'fraud_score': response.fraud_score, 'is_fraud': response.is_fraud}
