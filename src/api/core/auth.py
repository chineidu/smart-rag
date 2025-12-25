from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm.session import Session

from src import create_logger
from src.api.core.exceptions import HTTPError, UnauthorizedError
from src.config import app_config, app_settings
from src.db.crud import CRUDFactory
from src.db.models import DBUser, get_db
from src.schemas.types import RoleType
from src.schemas.user_schema import UserWithHashSchema

logger = create_logger(name="auth")
prefix: str = app_config.api_config.prefix
auth_prefix: str = app_config.api_config.auth_prefix


# =========== Configuration ===========
SECRET_KEY: str = app_settings.SECRET_KEY.get_secret_value()
ALGORITHM: str = app_settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES: int = app_settings.ACCESS_TOKEN_EXPIRE_MINUTES
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{auth_prefix}/token")
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl=f"{auth_prefix}/token", auto_error=False
)

# =========== Password hashing context ===========
# Using `scrypt` instead of `bcrypt` to avoid compatibility issues on macOS
pwd_context = CryptContext(schemes=["scrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# =========== JWT Token Management ===========
def create_access_token(
    data: dict[str, str], expires_delta: timedelta | None = None
) -> str:
    """Create a JWT access token."""
    to_encode: dict[str, Any] = data.copy()
    expire: datetime = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> UserWithHashSchema:
    """Get the current user from the JWT token."""
    crud_factory = CRUDFactory(db=db)
    try:
        payload: dict[str, Any] = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # type: ignore
        username: str | None = payload.get("sub")

        if username is None:
            raise UnauthorizedError(details="Could not validate credentials")

    except JWTError as e:
        raise UnauthorizedError(details="Could not validate credentials") from e

    db_user = crud_factory.get_user_by_username(username=username)
    if db_user is None:
        raise UnauthorizedError(details="User not found")
    user_schema = crud_factory.convert_userdb_to_schema(db_user)

    if user_schema is None:
        raise UnauthorizedError(details="User not found")

    if not user_schema.is_active:
        raise HTTPError(details="Inactive user")

    return user_schema


def authenticate_user(username: str, password: str, db: Session) -> DBUser | None:
    """Authenticate user with username and password."""
    crud_factory = CRUDFactory(db=db)
    user = crud_factory.get_user_by_username(username)
    if not user:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


async def get_current_active_user(
    current_user: UserWithHashSchema = Depends(get_current_user),
) -> UserWithHashSchema:  # noqa: B008
    """Get the current active user."""
    if not current_user.is_active:  # type: ignore
        raise HTTPError(details="Inactive user")
    return current_user


async def get_current_admin_user(
    current_user: UserWithHashSchema = Depends(get_current_active_user),
) -> UserWithHashSchema:  # noqa: B008
    """Get the current active admin user."""
    if RoleType.ADMIN not in current_user.roles:  # type: ignore
        raise HTTPError(
            details="Admin access required",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return current_user
