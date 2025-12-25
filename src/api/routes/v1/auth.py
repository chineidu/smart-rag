from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm.session import Session

from src import create_logger
from src.api.core.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
)
from src.api.core.exceptions import UnauthorizedError
from src.api.core.ratelimit import limiter
from src.api.core.reponses import MsgSpecJSONResponse
from src.config import app_config
from src.db.crud import CRUDFactory
from src.db.models import DBUser, get_db
from src.schemas.user_schema import UserCreateSchema, UserSchema, UserWithHashSchema

logger = create_logger(name="auth")
router = APIRouter(tags=["auth"], default_response_class=MsgSpecJSONResponse)
LIMIT_VALUE: int = app_config.api_config.ratelimit.login_rate


@router.post("/register", status_code=status.HTTP_201_CREATED)
@limiter.limit(f"{LIMIT_VALUE}/minute")
def register_user(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    user: UserCreateSchema,
    db: Session = Depends(get_db),
) -> UserSchema:
    """Register a new user.

    Checks that the provided username and email are unique, creates a new user
    record in the database, and returns the created user as a schema object.

    Parameters
    ----------
        user: UserCreateSchema
            Data required to create a new user (e.g. username, email, password).
        db: Session, optional
            Database session dependency used to query and persist user data
            (default is provided by dependency injection).

    Returns
    -------
        UserSchema
            Schema representation of the newly created user.
    """
    crud_factory = CRUDFactory(db=db)

    # Check if username exists
    db_user: DBUser | None = crud_factory.get_user_by_username(username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists. Please use a unique username",
        )
    # Check if username exists
    db_user = crud_factory.get_user_by_email(email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists. Please use a unique email",
        )

    # === Create new user ===
    hashed_password: str = get_password_hash(user.password.get_secret_value())  # type: ignore
    user_info = UserWithHashSchema(**user.model_dump(), hashed_password=hashed_password)

    new_user = crud_factory.create_user(user=user_info)
    return UserSchema(
        id=new_user.id,
        username=new_user.username,
        firstname=new_user.firstname,
        lastname=new_user.lastname,
        email=new_user.email,
        is_active=new_user.is_active,
        roles=[role.name for role in new_user.roles],
    )


@router.post("/token", status_code=status.HTTP_200_OK)
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def login_for_access_token(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """
    Authenticate a user and return an OAuth2 bearer access token.
    Validates credentials supplied via an OAuth2PasswordRequestForm (dependency-injected by FastAPI).
    On successful authentication a JWT access token is created and returned in the OAuth2 bearer format.

    Parameters
    ----------
        form_data: OAuth2PasswordRequestForm
            Dependency-injected form containing 'username' and 'password' fields.
            Provided by FastAPI via Depends().
        db: Session, optional
            Database session dependency used to query and persist user data
            (default is provided by dependency injection).

    Returns
    -------
        dict[str, str]

    """
    logger.info("Authenticating user...")
    user: DBUser | None = authenticate_user(
        username=form_data.username, password=form_data.password, db=db
    )
    if not user:
        raise UnauthorizedError(details="Incorrect username of password")
    logger.info(f"User {user.username!r} authenticated successfully.")
    access_token_expires: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def get_current_user(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    current_user: UserSchema = Depends(get_current_active_user),
) -> UserSchema:
    """
    Endpoint to get the current logged-in user. This endpoint is protected
    and requires a valid JWT token.

    Parameters
    ----------
        current_user: UserSchema
            The current logged-in user, provided by the dependency injection
            of get_current_active_user.

    Returns:
    -------
        UserSchema
            The current logged-in user's details.
    """

    return UserSchema(
        id=current_user.id,
        username=current_user.username,
        firstname=current_user.firstname,
        lastname=current_user.lastname,
        email=current_user.email,
        is_active=current_user.is_active,
        roles=current_user.roles,
    )
