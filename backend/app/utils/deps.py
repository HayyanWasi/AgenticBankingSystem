"""
Shared FastAPI dependency for authenticated endpoints.

Usage:
    from app.utils.deps import get_current_user, UserContext

    @router.post("/chat")
    async def chat(current_user: UserContext = Depends(get_current_user)):
        ...
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.database.engine import engine
from app.models import Account, User
from app.utils.auth import ALGORITHM, SECRET_KEY

_bearer = HTTPBearer(auto_error=False)


@dataclass
class UserContext:
    user_id: int
    full_name: str
    id_card_num: str
    account_number: Optional[str]  # Primary account number, None if not yet created
    balance: float
    is_admin: bool


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserContext:
    """
    Decode the Bearer JWT and return the authenticated user's context.
    Raises HTTP 401 if the token is absent or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[int] = int(payload.get("sub"))
        if user_id is None:
            raise ValueError("Missing subject claim")
    except (JWTError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with Session(engine) as db:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account not found.",
            )

        account = db.query(Account).filter(Account.user_id == user_id).first()

        return UserContext(
            user_id=user.user_id,
            full_name=user.full_name,
            id_card_num=user.id_card_num,
            account_number=account.account_number if account else None,
            balance=account.balance if account else user.balance,
            is_admin=user.is_admin,
        )

def require_admin(current_user: UserContext = Depends(get_current_user)) -> UserContext:
    """
    Dependency that enforces admin-only access.
    Raises HTTP 403 if the user is not an admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Admin privileges required.",
        )
    return current_user
