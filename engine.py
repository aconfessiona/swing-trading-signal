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
    rsi = 100 - 100/(1+rs)
    rsi = rsi.mask((avg_down == 0) & (avg_up > 0), 100)
    rsi = rsi.mask((avg_up == 0) & (avg_down > 0), 0)
    d['RSI14'] = rsi.fillna(50)
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


def version4(raw):
    """Version 4 indicators. Resistance uses only data available at each close."""
    d = signals(indicators(raw))
    c = d['Close']
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['MACD_hist'] = (ema12-ema26) - (ema12-ema26).ewm(span=9, adjust=False).mean()
    # Prior confirmed resistance, excluding today's high.
    d['prior_high_60'] = d['High'].shift(1).rolling(60, min_periods=30).max()
    # Rising parallel regression channel over prior 40 sessions: lower envelope and parallel upper envelope.
    window=40
    lows=d['Low'].to_numpy(dtype=float)
    highs=d['High'].to_numpy(dtype=float)
    lower=np.full(len(d),np.nan); upper=np.full(len(d),np.nan); slope=np.full(len(d),np.nan)
    x=np.arange(window,dtype=float)
    for i in range(window-1,len(d)):
        lo=lows[i-window+1:i+1]; hi=highs[i-window+1:i+1]
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()): continue
        m=np.polyfit(x,lo,1)[0]
        baseline=m*x
        intercept=np.min(lo-baseline)
        lower[i]=baseline[-1]+intercept
        upper[i]=lower[i]+np.max(hi-(baseline+intercept))
        slope[i]=m
    d['channel_lower']=lower
    d['channel_upper']=upper
    d['channel_rising']=slope>0
    # Weekly indicators: completed weeks only; map last completed week to each daily date.
    idx=pd.to_datetime(d.index)
    if idx.tz is not None: idx=idx.tz_localize(None)
    week=d.copy(); week.index=idx
    weekly=week.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'})
    weekly['week_upper']=weekly['Close'].rolling(20,min_periods=20).mean()+2*weekly['Close'].rolling(20,min_periods=20).std(ddof=0)
    weekly['week_prior_high']=weekly['High'].shift(1).rolling(26,min_periods=12).max()
    # A Friday's weekly value is only usable on the next trading day, not Friday intraday.
    completed=weekly[['week_upper','week_prior_high']].shift(1)
    daily_week=completed.reindex(idx,method='ffill')
    daily_week.index=d.index
    for col in daily_week: d[col]=daily_week[col]
    # Historical close-to-close volume profile: volume weighted by closing-price bins.
    # Approximation only; not an actual traded-by-price order book.
    vp=np.full(len(d),np.nan)
    if 'Volume' in d.columns:
        vols=pd.to_numeric(d['Volume'],errors='coerce').fillna(0).to_numpy(dtype=float)
        for i in range(80,len(d)):
            prices=c.iloc[i-80:i].to_numpy(dtype=float)
            weights=vols[i-80:i]
            if not np.isfinite(prices).all() or prices.min()<=0: continue
            edges=np.linspace(prices.min(),prices.max()*1.00001,21)
            hist,_=np.histogram(prices,bins=edges,weights=weights)
            centers=(edges[1:]+edges[:-1])/2
            above=centers>=float(c.iloc[i])*.98
            if above.any() and hist[above].max()>0:
                vp[i]=centers[np.where(above)[0][np.argmax(hist[above])]]
    d['volume_resistance']=vp
    levels=['BB_upper','channel_upper','prior_high_60','week_upper','week_prior_high','volume_resistance']
    d['resistance_near']=False
    d['resistance_count']=0
    for col in levels:
        near=(d['High']>=d[col]*.985)&(d['Low']<=d[col]*1.025)
        if col=='channel_upper': near=near & d['channel_rising']
        d['resistance_count']+=near.fillna(False).astype(int)
    d['resistance_near']=d['resistance_count']>0
    kdown=crossed_down(d['K5'],d['D5']) & (d['K5'].shift(1)>70)
    rsi_turn=(d['RSI14'].shift(1)>65)&(d['RSI14']<d['RSI14'].shift(1))
    macd_weak=(d['MACD_hist']<d['MACD_hist'].shift(1))&(d['MACD_hist'].shift(1)<d['MACD_hist'].shift(2))
    d['sell_half_condition']=d['resistance_near'] & (kdown|rsi_turn|macd_weak)
    # A two-day breakdown confirmation, with momentum weakness.
    d['trend_break_condition']=(c<d['MA20'])&(c.shift(1)<d['MA20'].shift(1)) & ((d['K10']<d['D10'])|(d['MACD_hist']<0))
    return d


def backtest_v4(d, strategy='A', fee_bps=15, max_hold=20, atr_multiple=2.5):
    """Close signal, next open execution; position state and half exits. Equity in initial capital units."""
    if strategy not in ('A','B'): raise ValueError('strategy must be A or B')
    d=d.dropna(subset=['MA60','K20','D20']).copy()
    cash=1.; units=0.; initial_units=0.; entry_cost=0.; entry_idx=None
    peak=0.; half_sold=False; pending=None; events=[]; curve=[]
    fee=fee_bps/10000
    for i,(date,r) in enumerate(d.iterrows()):
        op=float(r['Open']); cl=float(r['Close'])
        if pending and np.isfinite(op) and op>0:
            action,reason,signal_date=pending
            if action=='buy' and units==0:
                entry_cost=cash
                units=cash*(1-fee)/op; initial_units=units; cash=0.
                entry_idx=i; peak=cl; half_sold=False
                events.append({'signal_date':signal_date,'date':date,'action':'매수','reason':reason,'price':op,'fraction':1.0})
            elif action in ('half','exit') and units>0:
                sold=units*.5 if action=='half' and not half_sold else units
                cash+=sold*op*(1-fee); units-=sold
                events.append({'signal_date':signal_date,'date':date,'action':'1차 매도' if action=='half' and not half_sold else '전량 청산','reason':reason,'price':op,'fraction':sold/initial_units})
                if action=='half' and not half_sold: half_sold=True
                else: units=0.; half_sold=False; entry_idx=None
            pending=None
        curve.append((date,cash+units*cl))
        if i==len(d)-1: continue
        if units>0:
            peak=max(peak,cl)
            atr=float(r['ATR14']) if pd.notna(r['ATR14']) else np.nan
            stop=np.isfinite(atr) and cl<peak-atr_multiple*atr
            if stop: pending=('exit','ATR 보호 청산',date)
            elif bool(r['trend_break_condition']): pending=('exit','추세 이탈',date)
            elif i-entry_idx>=max_hold: pending=('exit','최대 보유기간',date)
            elif not half_sold and bool(r['sell_half_condition']): pending=('half','저항+모멘텀 약화',date)
        elif bool(r[strategy]): pending=('buy',f'{strategy} 매수',date)
    ev=pd.DataFrame(events)
    equity=pd.Series([v for _,v in curve],index=[t for t,_ in curve],name='equity',dtype=float)
    return ev,equity,units>0
