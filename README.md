# Swing Analysis 5.3.1 — Clean Sell Chart

Replace these four files together in the GitHub repository root: `app.py`, `engine.py`, `requirements.txt`, `README.md`.

Commit message: `Hide reactive sell candidates in Swing Analysis 5.3.1 chart`

## Changes from 5.3
- Hide reactive (반락 익절 후보) markers from the chart; **retain their calculation** in the engine and CSV diagnostic output.
- Keep proactive (선제 익절 후보) markers, which are already limited per swing by the 5.3 engine.
- Show only proactive candidates in the candidate summary table.
- Keep support watch, simulated entries, 30/30 partial sales, protective exits, moving averages, Bollinger bands, and resistance lines unchanged.

## Important
Proactive candidate markers are not actual executions. Simulated fills occur at next session open. Strategy performance is not validated. The engine's reactive signal can still influence its existing simulation; this update intentionally changes chart presentation, not trading rules.
