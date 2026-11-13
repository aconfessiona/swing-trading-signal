# Swing Analysis 5.0 C/D

## Upload files (exact filenames)
- `app.py` — Streamlit mobile-friendly dashboard
- `engine.py` — causal market regimes, structural support/resistance zones, deduplicated interest events, momentum
- `requirements.txt` — dependencies
- `README.md` — this guide

GitHub repo: `swing-trading-signal`, branch `main`. Upload/replace all four files together. Streamlit Cloud main file: `app.py`.

Suggested commit: `Add Swing Analysis 5.0 C/D structural zones and momentum dashboard`

## Usage

Run `pip install -r requirements.txt` then `streamlit run app.py`.
Enter ticker such as `373220.KS`, `005930.KS`, `AAPL`, or upload daily OHLCV CSV with `Date,Open,High,Low,Close,Volume`.

## Locked B-policy parameters
- 20-day moving average below for 3 sessions
- 1% confirmed swing-high breakout
- 10-session retest window
- 1.8 reward/risk minimum for future trade engine (not yet calculated)
- ADX(14) 18 reference threshold
- Downtrend and weak-uptrend prohibit new buys.

## Implemented in 5.0 C/D
- Retains A/B causal regime/buy-block engine.
- Confirms 5-bar pivots after two following bars, no lookahead in calculations.
- Groups historical confirmed daily pivot levels within 0.75 ATR.
- Identifies one representative watch event per support group (not a buy).
- Shows current support/resistance zones and separate ATR-normalized MACD histogram.
- Displays market status, detailed reasons, and downloadable daily diagnostics.
- Mobile responsive layout, reduced overlays and selectable indicators.

## Important limitations
- No actual buy/sell orders, position management, or backtest P&L in this version.
- Elliott wave labeling, weekly pivot confirmation, Fibonacci retracement, 30/30/40 exits are planned, not implemented.
- The support importance score is a heuristic ranking, not a calibrated probability.
- Horizontal current support/resistance shading is a *current-day visual guide*, not a historically available price series. The CSV contains daily as-of calculations.
- This is experimental technical analysis, not financial advice or a verified profitable strategy.
