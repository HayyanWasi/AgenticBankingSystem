from app.schemas.kyc_agent_schema import KYCState







# structured_kyc_llm = kyc_agent_result.with_structured_output(VerificationResult)
# def kyc_agent_results(state: KYCState):

#     last_message = state["messages"][-1]

#     if isinstance(last_message.content, list):
#         content = last_message.content[0]['text']
#     else:
#         content = last_message.content

#     response = structured_kyc_llm.invoke([
#         SystemMessage(content="Extract the KYC verification result from this message."),
#         HumanMessage(content=content)
#     ])

#     return {"kyc_score": response.kyc_score,
#              "verification_status": response.status,
#                "rejection_reason": response.rejection_reason,
#             }



#To reduce LLM calls costs
def kyc_agent_results(state: KYCState) -> dict:
    import json
    
    for message in reversed(state.get('messages', [])):
        if hasattr(message, 'tool_call_id'):
            try:
                data = json.loads(message.content)
                if 'score' in data:
                    score = data['score']
                    # Use status from tool if present, otherwise derive from score
                    status = data.get('status') or (
                        'approved' if score >= 0.8 else
                        'human_review' if score >= 0.4 else
                        'rejected'
                    )
                    return {
                        "kyc_score": score,
                        "verification_status": status,
                        "reject_reason": data.get('reason') or data.get('rejection_reason')
                    }
            except Exception:
                continue

    return {"verification_status": "failed", "reject_reason": "No evaluation found."}
