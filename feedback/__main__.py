"""
CLI entry point for the feedback package.

Usage:
    python -m feedback --out kto_pairs.jsonl
    python -m feedback --out kto_pairs.jsonl --brain FAST
    python -m feedback --out kto_pairs.jsonl --thumb 1
    python -m feedback --out kto_pairs.jsonl --min-chars 100
    python -m feedback --stats
"""
from __future__ import annotations

from pathlib import Path

import typer

from feedback.export import export_kto_jsonl, print_stats
from pg_settings import PgSettings

app = typer.Typer(add_completion=False, help="Sage Kaizen RLHF feedback export CLI")


@app.command()
def main(
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output JSONL path. Required unless --stats is set.",
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Print dataset statistics and exit.",
    ),
    brain: str | None = typer.Option(
        None,
        "--brain",
        help="Filter by brain: FAST or ARCHITECT.",
    ),
    thumb: int | None = typer.Option(
        None,
        "--thumb",
        help="Filter by thumb value: 1 (up) or -1 (down).",
    ),
    min_chars: int = typer.Option(
        50,
        "--min-chars",
        help="Skip responses shorter than N characters (default: 50).",
    ),
) -> None:
    cfg = PgSettings()

    if stats:
        print_stats(cfg.pg_dsn)
        raise typer.Exit()

    if out is None:
        typer.echo("ERROR: --out is required unless --stats is set.", err=True)
        raise typer.Exit(code=1)

    if brain and brain not in ("FAST", "ARCHITECT"):
        typer.echo("ERROR: --brain must be FAST or ARCHITECT.", err=True)
        raise typer.Exit(code=1)

    if thumb and thumb not in (1, -1):
        typer.echo("ERROR: --thumb must be 1 or -1.", err=True)
        raise typer.Exit(code=1)

    export_kto_jsonl(
        out,
        dsn=cfg.pg_dsn,
        brain=brain,
        thumb=thumb,
        min_chars=min_chars,
        verbose=True,
    )


if __name__ == "__main__":
    app()
