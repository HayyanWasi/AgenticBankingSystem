from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import loan, transfer, manager, privacy  # Consolidated routers
from app.api.v1 import user, auth
from app.api.v1.admin import router as admin_router
from app.api.v1.kyc import router as kyc_router
from app.database.engine import engine
from app.models import Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure all tables exist on startup
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="Agentic Banking API", version="1.0.0", lifespan=lifespan)

# 1. MUST ADD: CORS for Frontend Connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Register all Agents
app.include_router(kyc_router, prefix="/api/v1/kyc", tags=["KYC"])
app.include_router(loan.router, prefix="/api/v1/loan", tags=["Loans"])
app.include_router(transfer.router, prefix="/api/v1/transfer", tags=["Transfers"])
app.include_router(privacy.router, prefix="/api/v1/privacy", tags=["Support"])
app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["Admin"])
app.include_router(manager.router, prefix="/api/v1/manager", tags=["Orchestrator"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])