# Swing Analysis 5.1 C-MIXED

## GitHub upload: exact four files
- `app.py`
- `engine.py`
- `requirements.txt`
- `README.md`

Commit message: `Add Swing Analysis 5.1 mixed Bollinger sell diagnostics`

Streamlit entrypoint: `app.py`. Replace **all four** files in the same repository commit.

## Purpose
Research-only mixed Bollinger upper-band sell diagnostics: range-market upper-band rejection, uptrend confirmation, band-walk protection, short/mid/long daily confirmed-pivot resistance, and optional simulated held-position 30/30/40 exits. There is **no buy-signal engine**; selecting a hypothetical entry date does not mean a buy was justified.

## Technical rules
- Bollinger(20,2), MACD(12,26,9), ATR(14), confirmed five-candle pivots (2-bar confirmation delay).
- Sell watch: high reaches 99% of BB upper, or a confirmed resistance is approached.
- Range: sell confirmation needs BB re-entry or previous-low close break.
- Other regimes: score >=2 including BB re-entry or previous-low close break.
- Band walk: >=3 of last 5 bars have percent-B >=0.8, MA20 rising, regime UP, structure intact; release on 2 consecutive closes below MA20, pivot break, or 3 consecutive low-percent-B and fading momentum days.
- Resistance series: closest confirmed high above current close within 60, 180, 756 trading days. This is not volume profile, and weekly pivots are not implemented.
- Hypothetical entry at next open after selected date. Exits at next open after close confirmation; stops may fill intraday, with gap-aware pricing. The example uses fees and slippage but is not an execution simulator.

## Limitations
- Trading performance, parameters, and actual live quotes have not been verified.
- Backtesting entry selection and complete multi-timeframe resistance/volume profile require further work.
- Current resistance may jump as price passes a pivot; use date-specific series, not backfilled present levels.
