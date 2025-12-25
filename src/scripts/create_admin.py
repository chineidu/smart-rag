#!/usr/bin/env python3
"""
Bootstrap script to create the first admin user.

Usage: uv run -m src.scripts.create_admin --username admin --email admin@example.com --password mypassword
"""

import click

from src.api.core.auth import get_password_hash
from src.db.crud import CRUDFactory
from src.db.models import get_db_session
from src.schemas.types import RoleType
from src.schemas.user_schema import UserWithHashSchema


def create_admin_user(
    username: str, email: str, password: str, firstname: str = "", lastname: str = ""
) -> None:
    """Create the first admin user."""
    with get_db_session() as db:
        crud_factory = CRUDFactory(db=db)
        # Check if admin users already exist
        admin_role = crud_factory.get_role_by_name(name=RoleType.ADMIN)
        if admin_role and admin_role.users:
            print("❌ Admin users already exist. This script should only be run once.")
            return

        # Check if user already exists
        existing_user = crud_factory.get_user_by_username(username=username)
        if existing_user:
            print(f"❌ User '{username}' already exists.")
            return

        # Create admin user
        hashed_password = get_password_hash(password)
        user_data = UserWithHashSchema(
            username=username,
            email=email,
            firstname=firstname,
            lastname=lastname,
            hashed_password=hashed_password,
            is_active=True,
        )

        crud_factory.create_user(user=user_data)
        crud_factory.assign_role_to_user(username=username, role=RoleType.ADMIN)

        print("🎉 Admin user created successfully!")
        print(f"Username: {username}")
        print(f"Email: {email}")


@click.command(help="Create the first admin user")
@click.option("-u", "--username", required=True, help="Admin username")
@click.option("-e", "--email", required=True, help="Admin email")
@click.option(
    "--password",
    required=True,
    prompt=True,
    hide_input=True,
    confirmation_prompt=True,
    help="Admin password",
)
@click.option("-f", "--firstname", default="", help="Admin first name")
@click.option("-l", "--lastname", default="", help="Admin last name")
def main(
    username: str, email: str, password: str, firstname: str, lastname: str
) -> None:
    """Create the first admin user.

    Parameters
    ----------
    username : str
        The admin username
    email : str
        The admin email
    password : str
        The admin password
    firstname : str
        The admin first name
    lastname : str
        The admin last name

    Returns
    -------
    None
    """
    click.echo("🚀 Creating first admin user...")
    create_admin_user(
        username=username,
        email=email,
        password=password,
        firstname=firstname,
        lastname=lastname,
    )


if __name__ == "__main__":
    main()
