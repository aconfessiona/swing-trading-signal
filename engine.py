"""Swing 4.1 research engine. Daily close signals, next open simulated fills."""
import numpy as np
import pandas as pd


def crossed_up(a,b):
    return (a>b)&(a.shift(1)<=b.shift(1))

def crossed_down(a,b):
    return (a<b)&(a.shift(1)>=b.shift(1))

def indicators(raw):
    d=raw.copy().sort_index()
    d.columns=[str(c).strip().capitalize() for c in d.columns]
    for col in ['Open','High','Low','Close']:
        if col not in d: raise ValueError(f'Missing column: {col}')
        d[col]=pd.to_numeric(d[col],errors='coerce')
    d=d.dropna(subset=['Open','High','Low','Close'])
    d=d[(d[['Open','High','Low','Close']]>0).all(axis=1)]
    if 'Volume' not in d: d['Volume']=0.
    d['Volume']=pd.to_numeric(d['Volume'],errors='coerce').fillna(0)
    c=d.Close
    for n in (5,10,20,60): d[f'MA{n}']=c.rolling(n,min_periods=n).mean()
    sd=c.rolling(20,min_periods=20).std(ddof=0)
    d['BB_mid']=d.MA20; d['BB_upper']=d.MA20+2*sd; d['BB_lower']=d.MA20-2*sd
    for n,sk,ss in [(5,3,3),(10,6,6),(20,12,12)]:
        lo=d.Low.rolling(n,min_periods=n).min(); hi=d.High.rolling(n,min_periods=n).max()
        fast=(100*(c-lo)/(hi-lo).replace(0,np.nan)).clip(0,100)
        d[f'K{n}']=fast.rolling(sk,min_periods=sk).mean()
        d[f'D{n}']=d[f'K{n}'].rolling(ss,min_periods=ss).mean()
    prev=c.shift(1)
    tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    d['ATR14']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta=c.diff(); up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=up/down.replace(0,np.nan)
    d['RSI14']=(100-100/(1+rs)).mask((down==0)&(up>0),100).mask((up==0)&(down>0),0)
    d['MA20_gap_pct']=(c/d.MA20-1)*100
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26; d['MACD_hist']=macd-macd.ewm(span=9,adjust=False).mean()
    return d


def version41(raw):
    d=indicators(raw); c=d.Close
    # Candidate is a broader, visible setup; confirmation is a stricter entry signal.
    trend=(c>d.MA60*.97)&(d.MA20>=d.MA20.shift(5)*.985)
    pullback=(d.Low<=d.MA20*1.04)&(c>=d.MA20*.965)
    rebound=(d.K5>d.D5)&(d.K5>d.K5.shift(1))&(c>c.shift(1))
    stretch=(d.MA20_gap_pct<9)&(d.RSI14.between(30,72))
    d['A_candidate']=(trend&pullback&stretch&(d.K5>d.K5.shift(1))).fillna(False)
    d['A']=(d.A_candidate&rebound&(d.K10>=d.D10*.97)&(c>d.MA10)).fillna(False)
    oversold=((d.K5.rolling(5,min_periods=5).min()<25)|
              (d.Low.rolling(5,min_periods=5).min()<=d.BB_lower.rolling(5,min_periods=5).max()))
    d['B_candidate']=(oversold&(d.K5>d.K5.shift(1))&(d.RSI14<68)&(d.MA20_gap_pct<8)).fillna(False)
    d['B']=(d.B_candidate&rebound&(d.K10>=d.K10.shift(1))&(c>d.BB_lower)&(c>d.MA5)).fillna(False)
    # Levels known at today's close. Historical highs exclude today.
    d['prior_high_60']=d.High.shift(1).rolling(60,min_periods=30).max()
    w=40; lo=d.Low.to_numpy(float); hi=d.High.to_numpy(float)
    lower=np.full(len(d),np.nan); upper=np.full(len(d),np.nan); slope=np.full(len(d),np.nan)
    x=np.arange(w,dtype=float)
    for i in range(w-1,len(d)):
        a=lo[i-w+1:i+1]; b=hi[i-w+1:i+1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()): continue
        m=np.polyfit(x,a,1)[0]; baseline=m*x; intercept=np.min(a-baseline)
        lower[i]=baseline[-1]+intercept
        upper[i]=lower[i]+np.max(b-(baseline+intercept)); slope[i]=m
    d['channel_lower']=lower; d['channel_upper']=upper; d['channel_rising']=slope>0
    # Last completed weekly bar only: shift weekly series before mapping to daily.
    ix=pd.to_datetime(d.index)
    if ix.tz is not None: ix=ix.tz_localize(None)
    wk=d[['Open','High','Low','Close']].copy(); wk.index=ix
    weekly=wk.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'})
    weekly['week_upper']=weekly.Close.rolling(20,min_periods=20).mean()+2*weekly.Close.rolling(20,min_periods=20).std(ddof=0)
    weekly['week_prior_high']=weekly.High.shift(1).rolling(26,min_periods=12).max()
    mapped=weekly[['week_upper','week_prior_high']].shift(1).reindex(ix,method='ffill'); mapped.index=d.index
    for col in mapped: d[col]=mapped[col]
    # Approximate close-price volume histogram, not a true volume-at-price profile.
    vp=np.full(len(d),np.nan); volumes=d.Volume.to_numpy(float)
    for i in range(80,len(d)):
        prices=c.iloc[i-80:i].to_numpy(float); weights=volumes[i-80:i]
        if not np.isfinite(prices).all() or prices.min()<=0 or weights.sum()<=0: continue
        edges=np.linspace(prices.min(),prices.max()*1.00001,21)
        hist,_=np.histogram(prices,bins=edges,weights=weights)
        centers=(edges[:-1]+edges[1:])/2
        mask=centers>=float(c.iloc[i])*.98
        if mask.any() and hist[mask].max()>0:
            loc=np.flatnonzero(mask)[np.argmax(hist[mask])]; vp[i]=centers[loc]
    d['volume_resistance']=vp
    d['resistance_count']=0
    for col in ['BB_upper','channel_upper','prior_high_60','week_upper','week_prior_high','volume_resistance']:
        near=(d.High>=d[col]*.985)&(d.Low<=d[col]*1.025)
        if col=='channel_upper': near=near&d.channel_rising
        d['resistance_count']+=near.fillna(False).astype(int)
    d['resistance_near']=d.resistance_count>0
    kturn=crossed_down(d.K5,d.D5)&(d.K5.shift(1)>65)
    rturn=(d.RSI14.shift(1)>65)&(d.RSI14<d.RSI14.shift(1))
    mweak=(d.MACD_hist<d.MACD_hist.shift(1))&(d.MACD_hist.shift(1)<d.MACD_hist.shift(2))
    d['sell_half_condition']=(d.resistance_near&(kturn|rturn|mweak)).fillna(False)
    d['trend_break_condition']=((c<d.MA20)&(c.shift(1)<d.MA20.shift(1))&
       ((d.K10<d.D10)|(d.MACD_hist<0))).fillna(False)
    return d


def backtest_v41(d,strategy='A',fee_bps=15,max_hold=20,atr_multiple=2.5,min_profit_pct=2.0):
    """Single long position; signal on close, simulated fill at next open.
    Events include signal date AND execution date. Partial sale is half original units.
    """
    if strategy not in ('A','B'): raise ValueError('strategy must be A or B')
    d=d.dropna(subset=['MA60','K20','D20']).copy()
    cash=1.; units=0.; initial_units=0.; entry_price=np.nan; entry_i=None
    peak=0.; half_sold=False; pending=None; events=[]; curve=[]
    fee=fee_bps/10000
    for i,(date,r) in enumerate(d.iterrows()):
        op=float(r.Open); cl=float(r.Close)
        if pending is not None and np.isfinite(op) and op>0:
            action,reason,signal_date=pending
            if action=='buy' and units<=1e-12:
                units=cash*(1-fee)/op; initial_units=units; cash=0.
                entry_price=op; entry_i=i; peak=cl; half_sold=False
                events.append(dict(signal_date=signal_date,date=date,action='매수',reason=reason,price=op,fraction=1.))
            elif action in ('half','exit') and units>1e-12:
                partial=action=='half' and not half_sold
                sold=min(units,initial_units*.5) if partial else units
                cash+=sold*op*(1-fee); units-=sold
                events.append(dict(signal_date=signal_date,date=date,action='1차 매도' if partial else '전량 청산',reason=reason,price=op,fraction=sold/initial_units))
                if partial: half_sold=True
                else: units=0.; half_sold=False; entry_i=None
            pending=None
        curve.append((date,cash+units*cl))
        if i==len(d)-1: continue
        if units>1e-12:
            peak=max(peak,cl)
            atr=float(r.ATR14) if pd.notna(r.ATR14) else np.nan
            stop=np.isfinite(atr) and cl<peak-atr_multiple*atr
            if stop: pending=('exit','ATR 보호 청산',date)
            elif bool(r.trend_break_condition): pending=('exit','추세 이탈',date)
            elif i-entry_i+1>=max_hold: pending=('exit','최대 보유기간',date)
            elif (not half_sold and bool(r.sell_half_condition) and
                  cl>=entry_price*(1+min_profit_pct/100+2*fee)):
                pending=('half','저항·모멘텀 약화·수익 조건',date)
        elif bool(r[strategy]): pending=('buy',f'{strategy} 확정 매수',date)
    ev=pd.DataFrame(events,columns=['signal_date','date','action','reason','price','fraction'])
    eq=pd.Series([v for _,v in curve],index=[t for t,_ in curve],name='equity',dtype=float)
    return ev,eq,units>1e-12
