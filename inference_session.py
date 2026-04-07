"""
inference_session.py

Runtime connection state for all three llama-server endpoints.
Owns server lifecycle (start, stop, health, model discovery) so that
the UI and ChatService have no direct dependency on server_manager details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from openai_client import HttpTimeouts, discover_model_id, health_check
from server_manager import (
    ManagedServers,
    ensure_embed_running,
    ensure_q5_running,
    ensure_q6_running,
    ensure_summarizer_running,
    stop_server_on_port,
)


@dataclass
class InferenceSession:
    """
    Holds all runtime connection info for the three llama-server endpoints.

    Create one per Streamlit render cycle from sidebar URL inputs:

        session = InferenceSession.from_urls(q5_url, q6_url, embed_url, ...)

    The session is cheap to create (just a dataclass); ManagedServers is
    loaded from brains.yaml and owns server config and ports.
    """

    q5_url: str
    q6_url: str
    embed_url: str
    q5_model_id: str
    q6_model_id: str
    servers: ManagedServers
    # Optional CPU summarizer brain — auto-populated from brains.yaml when
    # the summarizer: section is uncommented; empty string = not configured.
    summarizer_url: str = ""
    summarizer_model_id: str = ""

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_urls(
        cls,
        q5_url: str,
        q6_url: str,
        embed_url: str,
        q5_model_id: str,
        q6_model_id: str,
    ) -> "InferenceSession":
        """
        Build a session from user-supplied base URLs.

        Summarizer URL and model ID are auto-populated from brains.yaml when
        the summarizer: section is uncommented.  When not configured, both
        fields default to "" and the system falls back to the FAST brain for
        search summarization.
        """
        servers = ManagedServers.from_yaml()
        summarizer_url = ""
        summarizer_model_id = ""
        if servers.summarizer is not None:
            summarizer_url = servers.summarizer.base_url
            summarizer_model_id = servers.summarizer.server.get("alias", "")
        return cls(
            q5_url=q5_url,
            q6_url=q6_url,
            embed_url=embed_url,
            q5_model_id=q5_model_id,
            q6_model_id=q6_model_id,
            servers=servers,
            summarizer_url=summarizer_url,
            summarizer_model_id=summarizer_model_id,
        )

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #

    def health_q5(self, timeouts: HttpTimeouts) -> Tuple[bool, str]:
        return health_check(self.q5_url, timeouts=timeouts)

    def health_q6(self, timeouts: HttpTimeouts) -> Tuple[bool, str]:
        return health_check(self.q6_url, timeouts=timeouts)

    # ------------------------------------------------------------------ #
    # Model discovery                                                      #
    # ------------------------------------------------------------------ #

    def discover_model_ids(
        self, timeouts: HttpTimeouts
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (q5_model_id, q6_model_id) as discovered from /v1/models."""
        mid5 = discover_model_id(self.q5_url, timeouts=timeouts)
        mid6 = discover_model_id(self.q6_url, timeouts=timeouts)
        return mid5, mid6

    # ------------------------------------------------------------------ #
    # Server lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def ensure_q5_ready(self) -> Tuple[bool, str]:
        """Start embed + Q5 if not running; block until ready or timeout."""
        return ensure_q5_running(self.servers)

    def ensure_q6_ready(self) -> Tuple[bool, str]:
        """Start Q6 if not running; block until ready or timeout."""
        return ensure_q6_running(self.servers)

    def ensure_summarizer_ready(self) -> Tuple[bool, str]:
        """
        Start the CPU summarizer brain if configured and not running.
        Returns (False, reason) when no summarizer is configured — not an error.
        """
        return ensure_summarizer_running(self.servers)

    def stop_all(self) -> Tuple[bool, bool, bool]:
        """Stop all three servers. Returns (ok_q5, ok_q6, ok_embed)."""
        ok5 = stop_server_on_port(self.servers.q5_port)
        ok6 = stop_server_on_port(self.servers.q6_port)
        ok_e = stop_server_on_port(self.servers.embed_port)
        return ok5, ok6, ok_e

    # ------------------------------------------------------------------ #
    # Brain → endpoint mapping                                             #
    # ------------------------------------------------------------------ #

    def url_for_brain(self, brain: str) -> str:
        """Return the base URL for the given brain name."""
        return self.q6_url if brain == "ARCHITECT" else self.q5_url

    def model_id_for_brain(self, brain: str) -> str:
        """Return the model ID for the given brain name."""
        return self.q6_model_id if brain == "ARCHITECT" else self.q5_model_id
