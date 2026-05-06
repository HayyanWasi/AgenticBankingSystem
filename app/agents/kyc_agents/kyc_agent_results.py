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
def kyc_agent_results(state: KYCState):
    import json
    
    # find tool message in state
    for message in reversed(state['messages']):
        if hasattr(message, 'name') and message.name == 'verify_customer_id':
            tool_result = json.loads(message.content)
            score = tool_result['score']
            
            if score >= 0.8:
                status = 'approved'
            elif score >= 0.4:
                status = 'human_review'
            else:
                status = 'rejected'
            
            rejection_reason = ""
            if 'reason' in tool_result:
                rejection_reason = tool_result['reason']
            elif not tool_result.get('name_match'):
                rejection_reason = "Name does not match ID records."
            elif not tool_result.get('not_expired'):
                rejection_reason = "ID card is expired."
                
            return {
                "kyc_score": score,
                "verification_status": status,
                "rejection_reason": rejection_reason
            }
