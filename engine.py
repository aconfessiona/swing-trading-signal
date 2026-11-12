"""Daily swing strategy research engine. No brokerage connection."""
import numpy as np
import pandas as pd


def indicators(raw):
    d = raw.copy().sort_index()
    for c in ['Open', 'High', 'Low', 'Close']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['Open','High','Low','Close'])
    close = d['Close']
    for n in (5, 10, 20, 60):
        d[f'MA{n}'] = close.rolling(n, min_periods=n).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    d['BB_mid'] = d['MA20']
    d['BB_upper'] = d['MA20'] + 2 * std
    d['BB_lower'] = d['MA20'] - 2 * std
    for n, smooth_k, smooth_d in [(5,3,3),(10,6,6),(20,12,12)]:
        low = d['Low'].rolling(n, min_periods=n).min()
        high = d['High'].rolling(n, min_periods=n).max()
        span = (high - low).replace(0, np.nan)
        fast = (100 * (close - low) / span).clip(0,100)
        d[f'K{n}'] = fast.rolling(smooth_k, min_periods=smooth_k).mean()
        d[f'D{n}'] = d[f'K{n}'].rolling(smooth_d, min_periods=smooth_d).mean()
    prev = close.shift(1)
    tr = pd.concat([(d['High']-d['Low']).abs(), (d['High']-prev).abs(), (d['Low']-prev).abs()], axis=1).max(axis=1)
    d['ATR14'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    delta = close.diff()
    avg_up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    d['RSI14'] = (100 - 100/(1+rs)).fillna(50)
    d['MA20_gap_pct'] = (close / d['MA20'] - 1) * 100
    return d


def crossed_up(k, sig):
    return (k > sig) & (k.shift(1) <= sig.shift(1))


def crossed_down(k, sig):
    return (k < sig) & (k.shift(1) >= sig.shift(1))


def signals(d):
    """Research signals; thresholds are provisional and require out-of-sample testing."""
    d = d.copy()
    up5 = crossed_up(d.K5, d.D5)
    dn5 = crossed_down(d.K5, d.D5)
    dn10 = crossed_down(d.K10, d.D10)
    up10 = crossed_up(d.K10, d.D10)
    trend = (d.Close > d.MA60) & (d.MA20 > d.MA60) & (d.MA20 >= d.MA20.shift(5))
    # Require CURRENT proximity to MA20, not just a touch sometime during the past 5 bars.
    near_ma20 = (d.Low <= d.MA20 * 1.025) & (d.Close >= d.MA20 * .985)
    not_extended = (d.MA20_gap_pct <= 8) & (d.RSI14 < 70)
    # Avoid treating a very stretched breakout as a pullback entry.
    d['A'] = trend & near_ma20 & not_extended & up5 & (d.K10 >= d.D10) & (d.K20 >= d.D20)
    oversold = ((d.K5.rolling(5).min() < 20) | (d.Close.rolling(5).min() <= d.BB_lower.rolling(5).max()))
    d['B'] = (oversold & up5 & (d.Close > d.Close.shift(1))
              & (up10 | (d.K10 > d.D10)) & (d.Close > d.BB_lower)
              & (d.RSI14 < 65) & (d.MA20_gap_pct < 8))
    # Confirm MA20 weakness for 2 consecutive closes, with weak medium-term momentum.
    below20 = (d.Close < d.MA20) & (d.Close.shift(1) < d.MA20.shift(1))
    d['exit_break'] = below20 & (d.Close < d.MA10) & (d.K10 < d.D10)
    d['exit_momentum'] = dn5 & (d.K5 > 55) & (d.K10 < d.D10) & (d.Close < d.MA20)
    d['exit_trend'] = (d.Close < d.MA60) & (d.MA20 < d.MA20.shift(3))
    d['exit'] = d[['exit_break', 'exit_momentum', 'exit_trend']].any(axis=1)
    d['warning'] = dn5 | dn10 | (d.Close < d.MA10)
    for col in ['A', 'B', 'exit', 'warning']:
        d[col] = d[col].fillna(False).astype(bool)
    return d


def backtest(d, strategy='A', fee_bps=15, max_hold=20):
    """Signal on close of t, execute at next available open t+1; no same-day lookahead.
    Long-only, one position, full capital, roundtrip costs as fee_bps per side.
    Equity marked at closes; forced time exit next open after max_hold bars.
    """
    if strategy not in ('A','B'): raise ValueError('strategy must be A or B')
    d = d.dropna(subset=['MA60','K20','D20']).copy()
    if len(d)<3: return pd.DataFrame(), pd.Series(dtype=float)
    capital=1.0; units=0.; entry_price=0.; entry_date=None; entry_idx=0; peak_close=None
    trades=[]; curve=[]; pending=None
    for i, (date,row) in enumerate(d.iterrows()):
        op=float(row.Open); cl=float(row.Close)
        if pending == 'sell' and units:
            proceeds=units*op*(1-fee_bps/10000)
            trades.append({'entry_date':entry_date,'exit_date':date,'entry_price':entry_price,'exit_price':op,'return_pct':(proceeds/(units*entry_price/(1+fee_bps/10000))-1)*100,'holding_bars':i-entry_idx,'exit_reason':pending_reason})
            capital=proceeds; units=0.; entry_date=None; peak_close=None
        elif pending == 'buy' and not units:
            entry_price=op; units=capital*(1-fee_bps/10000)/op
            entry_date=date; entry_idx=i; capital=0.; peak_close=cl
        pending=None
        equity=capital+units*cl
        curve.append((date,equity))
        if i==len(d)-1: continue
        if units:
            peak_close = max(peak_close, cl)
            # Signal at today's close; execute at next open. ATR stop is NOT a guaranteed stop fill.
            atr_exit = pd.notna(row.get('ATR14')) and cl < peak_close - 2.5 * float(row['ATR14'])
            if atr_exit: pending='sell'; pending_reason='ATR 추적 청산'
            elif bool(row['exit']): pending='sell'; pending_reason='기술적 청산'
            elif i-entry_idx+1 >= max_hold: pending='sell'; pending_reason='최대 보유기간'
        elif bool(row[strategy]): pending='buy'
    # Mark open position to last close, without inventing a completed trade.
    return pd.DataFrame(trades), pd.Series(dict(curve),name='equity',dtype=float)


def summary(trades,equity):
    if equity.empty: return {'total':0.,'mdd':0.,'count':0,'win':0.,'open_position':False}
    dd=equity/equity.cummax()-1
    return {'total':(equity.iloc[-1]-1)*100,'mdd':dd.min()*100,'count':len(trades),
            'win':(trades.return_pct.gt(0).mean()*100 if len(trades) else 0.)}
