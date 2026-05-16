from fastapi import APIRouter
from app.agents.bank_manager.agent import master_graph

router = APIRouter()

@router.get("/policy-search")
async def search_policy(query: str):
    # Direct tool access for a quick search without the full agent overhead
    from app.agents.privacy_policy_agent.pipeline import RAG
    rag = RAG()
    results = rag.vector_store.similarity_search(query, k=2)
    return {"results": [r.page_content for r in results]}