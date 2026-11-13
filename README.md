# Swing Analysis 5.3 — C Mixed BB Sell Opportunities

## Upload to GitHub
Replace all four files in the repository root: `app.py`, `engine.py`, `requirements.txt`, `README.md`.
Commit: `Add Swing Analysis 5.3 deduplicated sell opportunities and next-open fills`

## Changes
- Independent sell-opportunity markers across the entire chart, even before a selected virtual entry.
- Proactive Bollinger upper-band/RSI/resistance and reactive reversal candidates.
- Strong-trend band-walk protection and per-wave deduplication.
- Simulated 30% / 30% / remaining 40% exits, based on next-day open and protective stops.
- Diagnostic table of candidate dates and reasons.

## Cautions
Signals are exploratory and not validated for profitability. A virtual entry is not a confirmed buy signal. Resistance is based on confirmed daily pivots, not true volume-profile or weekly resistance. OHLC daily data cannot establish exact intraday sequence if stop and other prices are crossed in the same bar; protective stop takes priority. No future bars are used in signal calculations.
