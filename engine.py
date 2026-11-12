"""Daily swing research engine; signals at close, orders simulated next open."""
import numpy as np
import pandas as pd


def crossed_up(a,b):
    return (a>b)&(a.shift(1)<=b.shift(1))

def crossed_down(a,b):
    return (a<b)&(a.shift(1)>=b.shift(1))

def indicators(raw):
    d=raw.copy()
    d.columns=[str(x).strip().capitalize() for x in d.columns]
    if not {'Open','High','Low','Close'}.issubset(d.columns):
        raise ValueError('Open, High, Low, Close 열이 필요합니다.')
    d.index=pd.to_datetime(d.index)
    if d.index.tz is not None: d.index=d.index.tz_localize(None)
    d=d.sort_index(); d=d[~d.index.duplicated(keep='last')]
    for col in ['Open','High','Low','Close','Volume']:
        if col in d: d[col]=pd.to_numeric(d[col],errors='coerce')
    d=d.dropna(subset=['Open','High','Low','Close'])
    d=d[(d[['Open','High','Low','Close']]>0).all(axis=1)]
    c=d.Close
    for n in (5,10,20,60):d[f'MA{n}']=c.rolling(n,min_periods=n).mean()
    std=c.rolling(20,min_periods=20).std(ddof=0)
    d['BB_mid']=d.MA20;d['BB_upper']=d.MA20+2*std;d['BB_lower']=d.MA20-2*std
    for n,sk,sd in ((5,3,3),(10,6,6),(20,12,12)):
        low=d.Low.rolling(n,min_periods=n).min();high=d.High.rolling(n,min_periods=n).max()
        fast=(100*(c-low)/(high-low).replace(0,np.nan)).clip(0,100)
        d[f'K{n}']=fast.rolling(sk,min_periods=sk).mean()
        d[f'D{n}']=d[f'K{n}'].rolling(sd,min_periods=sd).mean()
    prev=c.shift()
    tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    d['ATR14']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta=c.diff();up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi=100-100/(1+up/down.replace(0,np.nan))
    d['RSI14']=rsi.mask((down==0)&(up>0),100).mask((up==0)&(down>0),0).fillna(50)
    d['MA20_gap_pct']=(c/d.MA20-1)*100
    return d

def signals(d):
    d=d.copy();c=d.Close
    up5=crossed_up(d.K5,d.D5);up10=crossed_up(d.K10,d.D10)
    trend=(c>d.MA60)&(d.MA20>d.MA60)&(d.MA20>=d.MA20.shift(5))
    near=(d.Low<=d.MA20*1.025)&(c>=d.MA20*.985)
    d['A']=(trend&near&(d.MA20_gap_pct<=8)&(d.RSI14<70)&up5&(d.K10>=d.D10)&(d.K20>=d.D20)).fillna(False)
    oversold=(d.K5.shift(1).rolling(5).min()<20)|(d.Low.shift(1).rolling(5).min()<=d.BB_lower.shift(1).rolling(5).max())
    d['B']=(oversold&up5&(c>c.shift())&(up10|(d.K10>d.D10))&(c>d.BB_lower)&(d.RSI14<65)&(d.MA20_gap_pct<8)).fillna(False)
    return d

def version4(raw):
    d=signals(indicators(raw));c=d.Close
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    d['MACD_hist']=macd-macd.ewm(span=9,adjust=False).mean()
    d['prior_high_60']=d.High.shift().rolling(60,min_periods=30).max()
    # Historical regression channel: all current-bar channel values are evaluated only at close.
    window=40;low=d.Low.to_numpy(float);high=d.High.to_numpy(float)
    lower=np.full(len(d),np.nan);upper=lower.copy();slope=lower.copy();x=np.arange(window,dtype=float)
    for i in range(window-1,len(d)):
        lo=low[i-window+1:i+1];hi=high[i-window+1:i+1]
        if not(np.isfinite(lo).all() and np.isfinite(hi).all()):continue
        m=np.polyfit(x,lo,1)[0];base=m*x;intercept=np.min(lo-base)
        lower[i]=base[-1]+intercept;upper[i]=lower[i]+np.max(hi-(base+intercept));slope[i]=m
    d['channel_lower']=lower;d['channel_upper']=upper;d['channel_rising']=slope>0
    # Weekly values from prior completed week only, even on Friday.
    weekly=d.resample('W-FRI').agg({'High':'max','Close':'last'})
    wstd=weekly.Close.rolling(20,min_periods=20).std(ddof=0)
    weekly['week_upper']=weekly.Close.rolling(20,min_periods=20).mean()+2*wstd
    weekly['week_prior_high']=weekly.High.shift().rolling(26,min_periods=12).max()
    # Label weekly bars at their next Saturday (00:00), so Friday's bar cannot see itself.
    known=weekly[['week_upper','week_prior_high']].copy()
    known.index=known.index+pd.Timedelta(days=1)
    aligned=known.reindex(d.index,method='ffill')
    for col in aligned:d[col]=aligned[col]
    vp=np.full(len(d),np.nan)
    if 'Volume' in d:
        vol=d.Volume.fillna(0).clip(lower=0).to_numpy(float)
        for i in range(80,len(d)):
            prices=c.iloc[i-80:i].to_numpy(float);weights=vol[i-80:i]
            if not np.isfinite(prices).all() or prices.min()<=0 or weights.sum()<=0:continue
            edges=np.linspace(prices.min(),prices.max()*1.00001,21)
            hist,_=np.histogram(prices,bins=edges,weights=weights)
            centers=(edges[1:]+edges[:-1])/2;above=centers>=float(c.iloc[i])*.98
            if above.any() and hist[above].max()>0:
                candidates=np.flatnonzero(above);vp[i]=centers[candidates[np.argmax(hist[above])]]
    d['volume_resistance']=vp
    levels=['BB_upper','channel_upper','prior_high_60','week_upper','week_prior_high','volume_resistance']
    d['resistance_count']=0
    for col in levels:
        # A resistance must be near the current trading range, not arbitrarily far away.
        near=(d.High>=d[col]*.985)&(d.Low<=d[col]*1.015)
        if col=='channel_upper':near=near&d.channel_rising
        d['resistance_count']+=near.fillna(False).astype(int)
    d['resistance_near']=d.resistance_count>0
    kdown=crossed_down(d.K5,d.D5)&(d.K5.shift()>70)
    rsi_turn=(d.RSI14.shift()>65)&(d.RSI14<d.RSI14.shift())
    macd_weak=(d.MACD_hist<d.MACD_hist.shift())&(d.MACD_hist.shift()<d.MACD_hist.shift(2))
    # Two resistance confirmations OR one resistance with momentum deterioration.
    d['sell_half_condition']=((d.resistance_count>=2)|((d.resistance_count>=1)&(kdown|rsi_turn|macd_weak))).fillna(False)
    d['trend_break_condition']=((c<d.MA20)&(c.shift()<d.MA20.shift())&((d.K10<d.D10)|(d.MACD_hist<0))).fillna(False)
    return d

def backtest_v4(d,strategy='A',fee_bps=15,max_hold=20,atr_multiple=2.5):
    if strategy not in ('A','B'):raise ValueError('strategy must be A or B')
    d=d.dropna(subset=['MA60','K20','D20']).copy()
    fee=fee_bps/10000;cash=1.;units=0.;initial_units=0.;entry_idx=None
    peak=0.;half_sold=False;pending=None;events=[];curve=[]
    for i,(date,r) in enumerate(d.iterrows()):
        op=float(r.Open);cl=float(r.Close)
        if pending is not None and np.isfinite(op) and op>0:
            action,reason,signal_date=pending
            if action=='buy' and units==0:
                units=cash*(1-fee)/op;initial_units=units;cash=0.;entry_idx=i;peak=cl;half_sold=False
                events.append(dict(signal_date=signal_date,date=date,action='매수',reason=reason,price=op,fraction=1.))
            elif action in ('half','exit') and units>0:
                is_half=action=='half' and not half_sold
                sold=units*.5 if is_half else units
                cash+=sold*op*(1-fee);units-=sold
                events.append(dict(signal_date=signal_date,date=date,action='1차 매도' if is_half else '전량 청산',reason=reason,price=op,fraction=sold/initial_units))
                if is_half:half_sold=True
                else:units=0.;half_sold=False;entry_idx=None
            pending=None
        curve.append((date,cash+units*cl))
        if i==len(d)-1:continue
        if units>0:
            peak=max(peak,cl)
            atr=float(r.ATR14) if pd.notna(r.ATR14) else np.nan
            stop=np.isfinite(atr) and cl<peak-atr_multiple*atr
            if stop:pending=('exit','ATR 보호 청산',date)
            elif bool(r.trend_break_condition):pending=('exit','추세 이탈',date)
            elif i-entry_idx>=max_hold:pending=('exit','최대 보유기간',date)
            elif not half_sold and bool(r.sell_half_condition):pending=('half','저항+모멘텀 약화',date)
        elif bool(r[strategy]):pending=('buy',f'{strategy} 매수',date)
    ev=pd.DataFrame(events,columns=['signal_date','date','action','reason','price','fraction'])
    equity=pd.Series([v for _,v in curve],index=[t for t,_ in curve],name='equity',dtype=float)
    return ev,equity,units>0
