"""Daily-bar swing signal engine. Signals at close; execution next open."""
import numpy as np
import pandas as pd


def prepare(raw):
    df = raw.copy()
    df.columns = [str(x).strip().lower() for x in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    needed = ['open','high','low','close','volume']
    missing = [x for x in needed if x not in df]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}. Date,Open,High,Low,Close,Volume 필요")
    df = df[needed].copy().sort_index()
    df = df.loc[~df.index.duplicated(keep='last')]
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=needed)
    df = df[(df['close']>0)&(df['open']>0)&(df['high']>=df['low'])]
    if len(df)<120:
        raise ValueError('최소 120거래일 이상의 일봉 데이터가 필요합니다.')
    return df


def stoch(df, n, smooth_k, smooth_d):
    lo = df.low.rolling(n,min_periods=n).min()
    hi = df.high.rolling(n,min_periods=n).max()
    raw = 100*(df.close-lo)/(hi-lo).replace(0,np.nan)
    k = raw.rolling(smooth_k,min_periods=smooth_k).mean()
    d = k.rolling(smooth_d,min_periods=smooth_d).mean()
    return k,d


def indicators(df):
    d = prepare(df)
    for n in [5,10,20,60]:
        d[f'ma{n}'] = d.close.rolling(n,min_periods=n).mean()
    for n in [10,20]:
        sd = d.close.rolling(n,min_periods=n).std(ddof=0)
        d[f'bb{n}_upper'] = d[f'ma{n}']+2*sd
        d[f'bb{n}_lower'] = d[f'ma{n}']-2*sd
    for n,k,dn in [(5,3,3),(10,6,6),(20,12,12)]:
        d[f'k{n}'],d[f'd{n}'] = stoch(d,n,k,dn)
    pc = d.close.shift(1)
    tr = pd.concat([(d.high-d.low),(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d['atr'] = tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['vol_ratio'] = d.volume/d.volume.rolling(20,min_periods=20).mean().replace(0,np.nan)
    cross = (d.k5>d.d5)&(d.k5.shift(1)<=d.d5.shift(1))
    pull = ((d.low<=d.ma10*1.01)|(d.low<=d.bb10_lower*1.02)).rolling(5,min_periods=5).max().fillna(0).astype(bool)
    a = (d.ma20>d.ma60)&(d.ma20>d.ma20.shift(5))&pull&cross&(d.k10>d.k10.shift(1))&(d.close>d.ma5)
    dipped = (d.close<d.bb10_lower).rolling(3,min_periods=3).max().shift(1).fillna(0).astype(bool)
    b = dipped&(d.close>d.bb10_lower)&cross&(d.k5<=30)&(d.k10>d.k10.shift(1))&(d.close>d.close.shift(1))
    d['signal'] = np.select([a,b],['A','B'],default='')
    # Ignore warm-up; all signals refer to completed daily candles.
    d.iloc[:80,d.columns.get_loc('signal')] = ''
    return d


def backtest(d, initial=10_000_000, risk_pct=1.0, fee_bps=15, max_position_pct=30):
    """One position at a time. Signals close t, buy open t+1; stop/target intraday.
    On ambiguous same-day stop/target, stop takes priority. Gap stops fill at open.
    Exit at next open after close-based exit signal. No margin/shorting.
    """
    cash=float(initial)
    qty=0; entry=stop=target=0.; strategy=''; entry_date=None; held=0
    trades=[]; equity=[]; pending=None; fee=fee_bps/10000
    for i,(dt,r) in enumerate(d.iterrows()):
        op,hi,lo,cl = (float(r[x]) for x in ['open','high','low','close'])
        if pending is not None:
            action=pending;pending=None
            if action['type']=='exit' and qty:
                gross=qty*op; cash+=gross*(1-fee)
                trades.append(dict(strategy=strategy,entry_date=entry_date,exit_date=dt,entry=entry,exit=op,qty=qty,return_pct=(op*(1-fee)/(entry*(1+fee))-1)*100,reason=action['reason']))
                qty=0
            elif action['type']=='buy' and not qty:
                atr=action['atr']; refstop=action['stop']; strat=action['strategy']
                if np.isfinite(atr) and atr>0 and np.isfinite(refstop):
                    # Avoid buying when a large overnight gap invalidates risk plan.
                    if op<=action['close']+atr and op>refstop:
                        stop=refstop
                        unitrisk=op-stop
                        riskbudget=cash*risk_pct/100
                        capbudget=cash*max_position_pct/100
                        n=int(min(riskbudget/unitrisk,capbudget/(op*(1+fee))))
                        if n>=1:
                            qty=n; entry=op; strategy=strat; entry_date=dt; held=0
                            cash-=qty*op*(1+fee)
                            target=entry+2*unitrisk if strat=='A' else max(entry+unitrisk,action['ma10'])
        if qty:
            held+=1
            # Gap through stop/target is filled at opening price, not trigger price.
            if op<=stop:
                px=op;reason='손절 갭'
            elif op>=target:
                px=op;reason='목표가 갭'
            elif lo<=stop:
                px=stop;reason='손절'
            elif hi>=target:
                px=target;reason='목표가'
            else:
                px=None;reason=''
            if px is not None:
                cash+=qty*px*(1-fee)
                trades.append(dict(strategy=strategy,entry_date=entry_date,exit_date=dt,entry=entry,exit=px,qty=qty,return_pct=(px*(1-fee)/(entry*(1+fee))-1)*100,reason=reason))
                qty=0
            elif pending is None:
                if strategy=='A' and (cl<r.ma10 or held>=20):
                    pending={'type':'exit','reason':'10일선 이탈/기간 청산'}
                elif strategy=='B' and (held>=10 or (r.k5<r.d5 and d.k5.iloc[i-1]>=d.d5.iloc[i-1])):
                    pending={'type':'exit','reason':'모멘텀/기간 청산'}
        if not qty and pending is None and i<len(d)-1 and r.signal in ('A','B'):
            atr=float(r.atr)
            if np.isfinite(atr) and atr>0:
                if r.signal=='A':
                    recent=float(d.low.iloc[max(0,i-9):i+1].min())
                    # Closer stop of two candidates, must be below signal close.
                    refstop=max(recent-0.1*atr,cl-2*atr)
                else:
                    recent=float(d.low.iloc[max(0,i-4):i+1].min())
                    refstop=max(recent-0.1*atr,cl-1.5*atr)
                if refstop<cl:
                    pending={'type':'buy','strategy':r.signal,'atr':atr,'stop':refstop,'close':cl,'ma10':float(r.ma10)}
        equity.append({'date':dt,'equity':cash+qty*cl})
    eq=pd.DataFrame(equity).set_index('date')
    t=pd.DataFrame(trades)
    if len(t):
        t['entry_date']=pd.to_datetime(t.entry_date)
        t['exit_date']=pd.to_datetime(t.exit_date)
    return t,eq


def stats(trades,eq,initial):
    end=float(eq.equity.iloc[-1]); dd=eq.equity/eq.equity.cummax()-1
    return {'수익률(%)':(end/initial-1)*100,'최대낙폭(%)':dd.min()*100,
            '완료 거래 수':len(trades),'승률(%)':float((trades.return_pct>0).mean()*100) if len(trades) else 0,
            '최종 평가액(원)':end}
