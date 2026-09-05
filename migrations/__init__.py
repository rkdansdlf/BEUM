"""Database migrations for the BEUM receiver server."""

from .runner import migrate_database

__all__ = ["migrate_database"]
