"""Swing 4.3 research engine. Completed daily/weekly/monthly bars only. No orders."""
import numpy as np
import pandas as pd

def cross_up(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
def stoch(d,n=20,sk=12,sd=12):
    lo=d.Low.rolling(n).min(); hi=d.High.rolling(n).max()
    fast=(100*(d.Close-lo)/(hi-lo).replace(0,np.nan)).clip(0,100)
    k=fast.rolling(sk).mean(); return k,k.rolling(sd).mean()
def higher_low(d):
    # confirmed pivot low: two bars either side, confirmation delayed 2 bars
    low=d.Low
    pivot=(low<low.shift(1))&(low<=low.shift(2))&(low<low.shift(-1))&(low<=low.shift(-2))
    confirmed=low.where(pivot).shift(2)
    prev=confirmed.ffill().shift(1)
    return confirmed.notna() & (confirmed>prev),confirmed.ffill()
def prepare(raw):
    d=raw.copy().sort_index(); d.columns=[str(c).split('_')[0].capitalize() for c in d.columns]
    for c in ['Open','High','Low','Close']:
        if c not in d: raise ValueError(f'Missing {c}')
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['Volume']=pd.to_numeric(d['Volume'],errors='coerce') if 'Volume' in d else np.nan
    d=d.dropna(subset=['Open','High','Low','Close']); d=d[~d.index.duplicated(keep='last')]
    c=d.Close
    for n in [5,10,20,60]: d[f'MA{n}']=c.rolling(n).mean()
    std=c.rolling(20).std(ddof=0);d['BB_upper']=d.MA20+2*std;d['BB_lower']=d.MA20-2*std
    for n,sk,sd in [(5,3,3),(10,6,6),(20,12,12)]:d[f'K{n}'],d[f'D{n}']=stoch(d,n,sk,sd)
    prev=c.shift();tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    d['ATR']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['vol_ratio']=d.Volume/d.Volume.shift(1).rolling(20,min_periods=15).mean().replace(0,np.nan)
    d['rsi_up']=c.diff().clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    down=(-c.diff().clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d['RSI']=100-100/(1+d.rsi_up/down.replace(0,np.nan));d['RSI']=d.RSI.fillna(50)
    d['prior_high']=d.High.shift(1).rolling(60,min_periods=30).max()
    d['prior_low']=d.Low.shift(1).rolling(30,min_periods=20).min()
    d['higher_low'],d['pivot_support']=higher_low(d)
    idx=pd.DatetimeIndex(d.index); idx=idx.tz_localize(None) if idx.tz is not None else idx
    z=d.copy();z.index=idx
    # Shift by one completed period: even current Friday/month-end cannot use unfinished higher timeframe.
    for rule,prefix in [('W-FRI','w'),('ME','m')]:
        h=z.resample(rule).agg({'Open':'first','High':'max','Low':'min','Close':'last'})
        h['K'],h['D']=stoch(h)
        vals=h[['K','D']].shift(1).rename(columns={'K':f'{prefix}K','D':f'{prefix}D'})
        aligned=vals.reindex(idx,method='ffill');aligned.index=d.index
        d[f'{prefix}K']=aligned[f'{prefix}K'];d[f'{prefix}D']=aligned[f'{prefix}D']
    return d

def strategy(raw, min_rr=1.5, volume_factor=1.3):
    d=prepare(raw); c=d.Close
    bull=(c>d.Open)&((c-d.Low)/(d.High-d.Low).replace(0,np.nan)>.60)
    volume=(d.vol_ratio>=volume_factor)
    recent_lower=(d.Low.shift(1).rolling(5).min()<=d.BB_lower.shift(1).rolling(5).max()*1.015)
    rebound=(c>d.BB_lower)&(c>c.shift(1))
    d['A_lower']=recent_lower&rebound&bull&volume
    # higher low confirmed two bars later; trigger only if bullish confirmation occurs recently
    d['B_higher_low']=d.higher_low.rolling(4).max().fillna(0).astype(bool)&bull&(c>c.shift(1))&volume
    rising=(d.MA20>d.MA20.shift(5))&(d.MA20>d.MA60)
    pullback=(d.Low.shift(1).rolling(4).min()<=d.MA20.shift(1).rolling(4).max()*1.02)
    d['C_pullback']=rising&pullback&bull&(c>d.MA20)&(c<d.MA20*1.055)&(d.vol_ratio>=1.05)
    range_high=d.High.shift(1).rolling(60,min_periods=60).max()
    range_low=d.Low.shift(1).rolling(60,min_periods=60).min()
    tight=(range_high/range_low-1)<.32
    d['D_base']=tight&(d.MA20>d.MA20.shift(5))&(c>d.MA20)&(c<range_high*1.025)&bull&volume
    d['pattern_count']=sum(d[k].astype(int) for k in ['A_lower','B_higher_low','C_pullback','D_base'])
    d['pattern']=np.select([d.A_lower,d.B_higher_low,d.C_pullback,d.D_base],['A 하단 반등','B 높아진 저점','C 상승 눌림','D 박스 탈출'],default='')
    # Momentum: weekly supportive OR rising, monthly only bonus (history often insufficient).
    d['weekly_ok']=(d.wK>d.wD)|(d.wK>d.wK.shift(5))
    d['monthly_ok']=(d.mK>d.mD)|(d.mK>d.mK.shift(20))
    d['overextended']=(c>d.MA20*1.08)|(c>d.BB_upper*1.015)|(d.RSI>70)|(c/c.shift(5)>1.13)
    # Resistance targets from PREVIOUS bars; if below price, use ATR projection only as conservative proxy.
    target=d.prior_high.where(d.prior_high>c*1.015,d.BB_upper.where(d.BB_upper>c*1.015,np.nan))
    stop=pd.concat([d.prior_low,d.MA20-1.5*d.ATR],axis=1).max(axis=1)
    risk=(c-stop).where(stop<c)
    d['RR']=(target-c)/risk.replace(0,np.nan)
    d['score']=d.pattern_count.clip(upper=1)*40+volume.astype(int)*20+bull.astype(int)*10+d.weekly_ok.fillna(False).astype(int)*15+d.monthly_ok.fillna(False).astype(int)*5+(d.RR>=min_rr).astype(int)*10
    d['candidate']=(d.pattern_count>0)&~d.overextended
    d['confirmed']=d.candidate&(d.score>=70)&d.weekly_ok.fillna(False)&(d.RR>=min_rr)&risk.notna()
    d['resistance']=target;d['initial_stop']=stop
    # de-duplicate candidate/confirmed markers within 8 bars, only first occurrence in a run
    for name in ['candidate','confirmed']:
        vals=d[name].fillna(False).to_numpy();shown=np.zeros(len(d),dtype=bool);last=-100
        for i,v in enumerate(vals):
            if v and i-last>=8: shown[i]=True;last=i
        d[name+'_display']=shown
    return d

def backtest(d,fee_bps=15,max_hold=30,atr_multiple=2.5,min_profit=.02):
    """Signals evaluated at close, executed next open. No retroactive fills."""
    fee=fee_bps/10000; cash=1.;units=0.;initial=0.;entry=0.;stop=np.nan;peak=0.;half=False;entry_i=0
    pending=None;events=[];curve=[]
    for i,(date,r) in enumerate(d.iterrows()):
        op=float(r.Open);cl=float(r.Close)
        if pending and np.isfinite(op) and op>0:
            action,signal_date,reason=pending
            if action=='buy' and units==0:
                units=cash*(1-fee)/op;initial=units;cash=0;entry=op;entry_i=i;peak=cl;half=False
                stop=float(d.loc[signal_date,'initial_stop']);events.append(dict(signal_date=signal_date,date=date,action='매수',price=op,reason=reason,fraction=1.))
            elif action in ('half','exit') and units>0:
                qty=units*.5 if action=='half' and not half else units
                cash+=qty*op*(1-fee);units-=qty
                events.append(dict(signal_date=signal_date,date=date,action='1차 50% 매도' if action=='half' and not half else '잔여/전량 청산',price=op,reason=reason,fraction=qty/initial))
                if action=='half' and not half:half=True
                else:units=0.;half=False
            pending=None
        curve.append((date,cash+units*cl))
        if i==len(d)-1:continue
        if units>0:
            peak=max(peak,cl)
            atr=float(r.ATR) if pd.notna(r.ATR) else 0.
            # protective stop can trigger regardless of profit; closing-price signal, next open execution
            broken=(cl<stop) or (cl<peak-atr_multiple*atr) or ((cl<r.MA20) and (d.Close.iloc[i-1]<d.MA20.iloc[i-1]) and (r.K10<r.D10))
            resistance=pd.notna(r.resistance) and r.High>=r.resistance*.985
            weakening=(r.K5<r.D5) or (r.RSI<d.RSI.iloc[i-1])
            if broken:pending=('exit',date,'손절/추세 이탈')
            elif i-entry_i>=max_hold:pending=('exit',date,'최대 보유기간')
            elif not half and resistance and weakening and cl>=entry*(1+min_profit):pending=('half',date,'저항+모멘텀 약화')
        elif bool(r.confirmed):pending=('buy',date,str(r.pattern))
    return pd.DataFrame(events),pd.Series(dict(curve),dtype=float,name='equity'),units>0
