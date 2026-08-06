"""
evals/ — model evaluation harness for Sage Kaizen.

Answers one question: **is a candidate model worth upgrading to, for THIS system?**
See Benchmarking_Kaizen_Models.md for the methodology and the decision rule.

Layers, cheapest first — a candidate failing one never reaches the next:

    1. gates.py    — hard pass/fail compatibility gates (minutes, no judgment)
    2. bench.py    — llama-bench performance measurement + run comparison
    3. golden.py   — frozen golden set mined from real logged turns
    4. scorers.py  — deterministic quality scorers (CJK, citations, <think>, length)

Nothing here imports Streamlit, spawns a brain, or touches the display GPU.
Import the submodules directly — ``from evals.gates import static_gates_fast``
— rather than relying on ``evals`` re-exporting them. Nothing is imported here
on purpose: `golden` reaches into `router` and `chat_service` for the slice
taxonomy, and a bare ``import evals`` should not drag the app in with it.
"""
from __future__ import annotations
