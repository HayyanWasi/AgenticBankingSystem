from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import kyc, loan, transfer, manager, privacy # Consolidated routers

app = FastAPI(title="Agentic Banking API", version="1.0.0")

# 1. MUST ADD: CORS for Frontend Connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Register all Agents
app.include_router(kyc.router, prefix="/api/v1/kyc", tags=["KYC"])
app.include_router(loan.router, prefix="/api/v1/loan", tags=["Loans"])
app.include_router(transfer.router, prefix="/api/v1/transfer", tags=["Transfers"])
app.include_router(privacy.router, prefix="/api/v1/privacy", tags=["Support"])
app.include_router(manager.router, prefix="/api/v1/manager", tags=["Orchestrator"])