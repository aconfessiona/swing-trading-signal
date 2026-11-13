# Swing Analysis 5.2 — C Mixed Early Exit

Four-file Streamlit application. Upload app.py, engine.py, requirements.txt, README.md together to GitHub repository root.

## Changes from 5.1
- Remove repetitive sell-watch circles from the chart. Sell watch remains available in diagnostics.
- Proactive first 30% exit candidate at Bollinger upper / resistance with RSI or structural confluence; stricter in protected band walks.
- Separate second exit candidate after a new price extension and subsequent reversal.
- Display only current three resistance levels, not historical stair-step traces. These are current levels and MUST NOT be interpreted as historical levels.
- Protective exit overrides partial exits; daily close signal -> next open simulation.

## Limitations
This is experimental, not a validated trading strategy. No real buy signals or real orders. The 30/30/40 simulation assumes a manually selected hypothetical entry. Structural resistance is based on confirmed daily pivot highs, not actual volume profile or weekly pivots. No claim of optimal peak timing. Review historical diagnostics and transaction costs before using for decisions.

## Install
`pip install -r requirements.txt`
`streamlit run app.py`

## Deploy
Replace all four files together, commit: `Add Swing Analysis 5.2 early exits and declutter sell markers`
