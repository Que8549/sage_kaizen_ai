from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class PgSettings(BaseSettings):
    """Shared PostgreSQL connection fields.

    Subclass this to inherit PG_* env-var bindings and ``pg_dsn``
    without repeating the field definitions.

    Values are populated in this order:
        1. .env file (project root)
        2. OS environment variables
        3. Default values defined below

    Example .env file (project root):

        PG_USER=sage
        PG_PASSWORD=YourRealPassword
        PG_DB=sage_kaizen
    """

    model_config = SettingsConfigDict(
        env_file=".env",
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
