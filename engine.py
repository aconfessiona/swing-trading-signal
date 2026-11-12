"""Swing 4.4 research engine. Daily close signals, next-session open simulated fills."""
import numpy as np
import pandas as pd


def _cross_up(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
def _cross_down(a,b): return (a<b)&(a.shift(1)>=b.shift(1))

def prepare(raw):
    d=raw.copy().sort_index()
    d.index=pd.to_datetime(d.index).tz_localize(None) if pd.to_datetime(d.index).tz is not None else pd.to_datetime(d.index)
    d=d[~d.index.duplicated(keep='last')]
    for c in ['Open','High','Low','Close']:
        if c not in d: raise ValueError(f'Missing column: {c}')
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['Volume']=pd.to_numeric(d['Volume'],errors='coerce').fillna(0) if 'Volume' in d else 0.
    d=d.dropna(subset=['Open','High','Low','Close'])
    d=d[(d[['Open','High','Low','Close']]>0).all(axis=1)]
    c=d.Close
    for n in [5,10,20,60]:d[f'MA{n}']=c.rolling(n,min_periods=n).mean()
    sd=c.rolling(20,min_periods=20).std(ddof=0)
    d['BB_upper']=d.MA20+2*sd;d['BB_lower']=d.MA20-2*sd
    d['VOL20']=d.Volume.shift(1).rolling(20,min_periods=15).mean()
    d['volume_ratio']=d.Volume/d.VOL20.replace(0,np.nan)
    prev=c.shift(1)
    tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    d['ATR14']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    change=c.diff();gain=change.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean();loss=(-change.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['RSI14']=(100-100/(1+gain/loss.replace(0,np.nan))).mask((loss==0)&(gain>0),100).mask((gain==0)&(loss>0),0).fillna(50)
    for n,k_s,d_s in [(5,3,3),(10,6,6),(20,12,12)]:
        lo=d.Low.rolling(n,min_periods=n).min();hi=d.High.rolling(n,min_periods=n).max()
        fast=(100*(c-lo)/(hi-lo).replace(0,np.nan)).clip(0,100)
        d[f'K{n}']=fast.rolling(k_s,min_periods=k_s).mean();d[f'D{n}']=d[f'K{n}'].rolling(d_s,min_periods=d_s).mean()
    # Prior-only pivots: a pivot is recognized after 3 subsequent bars.
    lo=d.Low.to_numpy();hi=d.High.to_numpy();pivot_low=np.full(len(d),np.nan);pivot_high=np.full(len(d),np.nan)
    for i in range(6,len(d)):
        j=i-3
        if lo[j]==np.min(lo[j-3:j+4]):pivot_low[i]=lo[j]
        if hi[j]==np.max(hi[j-3:j+4]):pivot_high[i]=hi[j]
    d['support']=pd.Series(pivot_low,index=d.index).ffill()
    d['resistance']=pd.Series(pivot_high,index=d.index).ffill()
    d['prior_low60']=d.Low.shift(1).rolling(60,min_periods=20).min()
    d['prior_high60']=d.High.shift(1).rolling(60,min_periods=20).max()
    # Two most recently confirmed pivot lows define a projected lower trendline.
    lower=np.full(len(d),np.nan);upper=np.full(len(d),np.nan)
    pivots=[]
    for i in range(len(d)):
        if np.isfinite(pivot_low[i]):pivots.append((i-3,pivot_low[i]))
        if len(pivots)>=2:
            (i1,v1),(i2,v2)=pivots[-2:]
            if i2>i1:
                lower[i]=v2+(v2-v1)/(i2-i1)*(i-i2)
                upper[i]=lower[i]+max(0,d.High.iloc[max(0,i-40):i+1].max()-d.Low.iloc[max(0,i-40):i+1].min())
    d['trend_support']=lower;d['trend_upper_approx']=upper
    # Fibonacci anchor: prior 60-session high/low (excludes current bar); only upward swings.
    rolling_low=d.Low.shift(1).rolling(60,min_periods=25).min();rolling_high=d.High.shift(1).rolling(60,min_periods=25).max()
    d['fib_low']=rolling_low;d['fib_high']=rolling_high
    for f in [.236,.382,.5,.618]:d[f'fib_{f}']=rolling_high-(rolling_high-rolling_low)*f
    # Completed weekly bars only. Shift before forward filling to daily timestamps.
    w=d[['Open','High','Low','Close']].resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'})
    w['weekly_prior_high']=w.High.shift(1).rolling(26,min_periods=8).max()
    w['weekly_bb_upper']=w.Close.rolling(20,min_periods=20).mean()+2*w.Close.rolling(20,min_periods=20).std(ddof=0)
    for col in ['weekly_prior_high','weekly_bb_upper']:
        d[col]=w[col].shift(1).reindex(d.index,method='ffill').to_numpy()
    # Confirmed support/reversal price action is primary; stochastic is informational only.
    atr=d.ATR14.fillna(c*.03)
    near=lambda level,tol: level.notna() & (d.Low<=level+tol) & (d.High>=level-tol)
    support_tests=[near(d.BB_lower,atr*.4),near(d.support,atr*.55),near(d.prior_low60,atr*.6),near(d.trend_support,atr*.6)]
    for f in [.236,.382,.5,.618]:support_tests.append(near(d[f'fib_{f}'],atr*.4))
    d['support_hits']=sum(s.astype(int) for s in support_tests)
    body=(d.Close-d.Open);span=(d.High-d.Low).replace(0,np.nan)
    bullish=(body>0)&((d.Close-d.Low)/span>.62)
    rejection=((d[['Open','Close']].min(axis=1)-d.Low)/span>.35)&(d.Close>d.Low+span*.55)
    d['reversal_candle']=(bullish|rejection)&(d.Close>=d.Close.shift(1)*.985)
    d['uptrend']=(d.MA20>d.MA60)&(d.MA20>d.MA20.shift(5))
    d['pullback']=d.uptrend & near(d.MA20,atr*.7) & (d.Close>d.MA60)
    d['breakout_retest']=(d.Close>d.resistance.shift(1)*.985)&(d.Low<=d.resistance.shift(1)*1.025)&(d.Close>d.Open)
    # Prevent a chased entry near resistance or after a fast spike.
    d['overextended']=(d.Close>d.MA20*1.085)|(c/c.shift(5)>1.13)|(d.RSI14>73)
    # Balanced entry: never buy merely on an oversold reading or a single support touch.
    # Confirmation requires a prior test, recovery above a recent high, and an improving base.
    recent_support=d.support_hits.shift(1).rolling(8,min_periods=1).max().fillna(0)>=1
    swing_low=d.Low.shift(1).rolling(12,min_periods=6).min()
    older_low=d.Low.shift(13).rolling(18,min_periods=8).min()
    stabilizing=(swing_low>=older_low*.975)&(c>c.shift(3))
    confirmation=(c>d.High.shift(1))&(c>d.Open)&(c>c.shift(2))
    recovery=(c>d.MA20)&(d.MA20>=d.MA20.shift(4)*.995)
    # Uptrend pullback: reclaim prior day's high near a rising MA20, without chasing.
    pullback_recent=(d.Low.shift(1).rolling(6,min_periods=3).min()<=d.MA20.shift(1).rolling(6,min_periods=3).max()*1.015)
    uptrend_pullback=d.uptrend&pullback_recent&confirmation&(c>d.MA20)&(c>d.MA60)
    # Base reversal: wait for a higher low AND an MA20 reclaim, not the first falling-knife bounce.
    base_reversal=recent_support&stabilizing&confirmation&recovery
    # Breakout-retest: resistance level must have been crossed earlier, then tested as support.
    level=d.resistance.shift(3)
    breakout_seen=(d.Close.shift(1).rolling(5,min_periods=2).max()>level*1.015)
    retest=breakout_seen&(d.Low<=level*1.025)&(d.Low>=level*.965)&confirmation&(c>level)&(c>d.MA20)
    d['buy_candidate']=((d.support_hits>=1)|d.pullback|d.breakout_retest)&~d.overextended
    d['buy_confirmed']=(base_reversal|uptrend_pullback|retest)&~d.overextended
    d['buy_score']=d.support_hits.clip(upper=4)*2+d.reversal_candle.astype(int)*2+d.pullback.astype(int)*2+d.breakout_retest.astype(int)*2+(d.volume_ratio>=1.3).astype(int)
    # Resistance proximity and bearish price reaction; no stochastic gating.
    resistance_levels=['BB_upper','resistance','prior_high60','weekly_prior_high','weekly_bb_upper','trend_upper_approx']
    d['resistance_hits']=sum(near(d[col],atr*.55).astype(int) for col in resistance_levels)
    # Profit-taking needs genuine reversal, not resistance proximity alone.
    upper_wick=(d.High-d[['Open','Close']].max(axis=1))/span
    rejection=(upper_wick>.35)&(d.Close<d.Open)&(d.Close<d.High-span*.45)
    failed_breakout=(d.High>=d.High.shift(1))&(d.Close<d.Close.shift(1))&(d.Close<d.Open)
    d['bearish_reaction']=(rejection|failed_breakout).fillna(False)
    d['sell_candidate']=d.resistance_hits>=1
    d['sell_confirmed']=d.sell_candidate&d.bearish_reaction
    d['trend_break']=(d.Close<d.MA20)&(d.Close.shift(1)<d.MA20.shift(1))
    for col in ['buy_candidate','buy_confirmed','sell_candidate','sell_confirmed','trend_break']:
        d[col]=d[col].fillna(False).astype(bool)
    return d


def simulate(d,fee_bps=15,atr_multiple=2.5,max_hold=40):
    """Long-only, one position at a time, 30/30/40. Next-open fills; no lookahead."""
    d=d.dropna(subset=['MA60','ATR14']).copy()
    fee=fee_bps/10000
    cash=1.;units=0.;initial=0.;entry_i=None;entry_price=0.;peak=0.
    stage=0;pending=None;events=[];equity=[];last_exit=-999;last_sell=-999
    cooldown=15
    for i,(date,r) in enumerate(d.iterrows()):
        op=float(r.Open);cl=float(r.Close)
        if pending is not None and np.isfinite(op) and op>0:
            action,reason,signal_date=pending
            if action=='buy' and units==0:
                units=cash*(1-fee)/op;initial=units;cash=0.
                entry_i=i;entry_price=op;peak=op;stage=0
                events.append(dict(signal_date=signal_date,date=date,action='매수',reason=reason,price=op,fraction=1.))
            elif action in ('sell1','sell2','exit') and units>0:
                fraction={'sell1':.3,'sell2':.3,'exit':1.}[action]
                sold=min(units,initial*fraction) if action!='exit' else units
                cash+=sold*op*(1-fee);units-=sold
                name={'sell1':'1차 30% 매도','sell2':'2차 30% 매도','exit':'잔여 청산'}[action]
                events.append(dict(signal_date=signal_date,date=date,action=name,reason=reason,price=op,fraction=sold/initial))
                last_sell=i
                if action=='sell1':stage=1
                elif action=='sell2':stage=2
                else:
                    units=0.;stage=0;entry_i=None;last_exit=i
            pending=None
        equity.append((date,cash+units*cl))
        if i==len(d)-1:continue
        if units>0:
            peak=max(peak,cl)
            atr=float(r.ATR14)
            # Risk management takes precedence. No claim that these are peak sells.
            hard_stop=cl<entry_price-2.0*atr
            trailing=cl<peak-atr_multiple*atr and peak>entry_price+1.5*atr
            if hard_stop:pending=('exit','초기 손절',date)
            elif trailing:pending=('exit','ATR 추적 손절',date)
            elif bool(r.trend_break) and (i-entry_i>=5) and cl<entry_price*.96:pending=('exit','추세 이탈·손실 제한',date)
            elif stage==0 and i-entry_i>=7 and cl>entry_price*1.065 and bool(r.sell_confirmed) and peak>entry_price*1.08:
                pending=('sell1','저항 확인·수익 구간',date)
            elif stage==1 and i-last_sell>=5 and cl>entry_price*1.11 and bool(r.sell_confirmed) and peak>entry_price*1.13:
                pending=('sell2','추가 저항 확인·수익 구간',date)
        elif i-last_exit>=cooldown and bool(r.buy_confirmed):
            pending=('buy','지지 구조·반등 확인',date)
    return pd.DataFrame(events,columns=['signal_date','date','action','reason','price','fraction']),pd.Series([v for _,v in equity],index=[t for t,_ in equity],name='equity',dtype=float),units>0
