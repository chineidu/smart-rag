"""
Database initialization utilities.
"""

from src import create_logger
from src.db.crud import create_role
from src.db.models import Base, get_db_pool
from src.schemas.types import RoleType
from src.schemas.user_schema import RoleSchema

logger = create_logger(name="db_init")


def init_db() -> None:
    """Initialize the database by creating tables and default roles.

    This function should be called once when the application starts.
    It creates all database tables and populates default roles if they don't exist.
    """
    db_pool = get_db_pool()
    # Create all tables in the database
    Base.metadata.create_all(db_pool.engine)
    logger.info("Database tables initialized")

    # Create default roles if they do not exist
    descriptions: list[str] = [
        "Administrator with full access",
        "Standard user with limited access",
        "Guest user with read-only access",
    ]

    with db_pool.get_session() as session:
        try:
            for role, desc in zip(RoleType, descriptions):
                create_role(db=session, role=RoleSchema(name=role, description=desc))
            logger.info("Default roles created successfully")
        except Exception as e:
            logger.error(f"Error creating default roles: {e}")
            raise e
