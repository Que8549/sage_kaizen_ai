"""
news/

Sage Kaizen Daily News Runtime.

Top-level package.  Sub-packages are imported lazily to avoid startup
overhead and to allow graceful degradation if optional deps (APScheduler,
yfinance) are not yet installed.
"""
