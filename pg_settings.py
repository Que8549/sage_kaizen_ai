from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to THIS FILE, not the process CWD.
#
# `env_file=".env"` is CWD-relative, so any process started from a directory
# other than the project root silently found no .env and fell back to the
# placeholder defaults below — connecting as my_user/my_pwd to my_db, which
# fails with an authentication error that names none of that.  Every ingest
# script is launched from the ingest project root, so this affected them all.
#
# pg_settings.py sits at the project root, so parent-of-this-file is always
# <project_root>.  The fix came from the sage_kaizen_ai_ingest copy of this
# module, which had it and was dead code (CLAUDE.md §13).


class PgSettings(BaseSettings):
    """Shared PostgreSQL connection fields.

    Subclass this to inherit PG_* env-var bindings and ``pg_dsn``
    without repeating the field definitions.

    Values are populated in this order:
        1. .env file (project root — resolved relative to this file)
        2. OS environment variables
        3. Default values defined below

    Example .env file (project root):

        PG_USER=sage
        PG_PASSWORD=YourRealPassword
        PG_DB=sage_kaizen
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    pg_user: str = "my_user"
    pg_password: str = "my_pwd"
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_db: str = "my_db"

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )
