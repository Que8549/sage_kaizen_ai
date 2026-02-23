from __future__ import annotations

from pg_settings import PgSettings


class FeedbackSettings(PgSettings):
    """
    Loaded from the same .env as RagSettings.
    PG_* keys are shared — the feedback schema lives in the same database.

    All fields (pg_user, pg_password, pg_host, pg_port, pg_db, pg_dsn)
    are inherited from PgSettings.
    """
