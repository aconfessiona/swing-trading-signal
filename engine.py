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
    # REGIME ENGINE: classify using only information available by this close.
    atr=d.ATR14.fillna(c*.025)
    near=lambda level,tol: level.notna() & (d.Low<=level+tol) & (d.High>=level-tol)
    body=d.Close-d.Open
    span=(d.High-d.Low).replace(0,np.nan)
    bullish=(body>0)&((d.Close-d.Low)/span>.65)
    rejection=((d[['Open','Close']].min(axis=1)-d.Low)/span>.35)&(d.Close>d.Low+span*.55)
    d['reversal_candle']=(bullish|rejection)&(c>c.shift(1)*.985)
    d['ma20_below_days']=(c<d.MA20).astype(int).groupby((c>=d.MA20).cumsum()).cumsum()
    # Prior-confirmed pivots only: never use a pivot until three bars have elapsed.
    recent_high=d.resistance.shift(1)
    recent_low=d.support.shift(1)
    failed_high=(d.High>=recent_high*.985)&(c<recent_high*.985)&(c<c.shift(1))
    # A breakout is recorded only after a completed close above prior resistance.
    breakout=(c>recent_high*1.01)&(c.shift(1)<=recent_high.shift(1)*1.01)
    breakout_level=recent_high.where(breakout).ffill()
    bars_since_breakout=pd.Series(np.arange(len(d)),index=d.index)-pd.Series(np.arange(len(d)),index=d.index).where(breakout).ffill()
    failed_breakout=(bars_since_breakout.between(1,8))&(c<breakout_level*.99)
    d['failed_high']=failed_high.fillna(False)
    d['failed_breakout']=failed_breakout.fillna(False)
    # Confirmed swing low breach is a structural break, not a routine MA20 touch.
    low_break=(c<recent_low*.985)&(c.shift(1)>=recent_low.shift(1)*.985)
    ma_weak=d.ma20_below_days>=3
    slope_down=d.MA20<d.MA20.shift(5)*.995
    ma_bear=(d.MA20<d.MA60)&slope_down
    d['risk_score']=(ma_weak.astype(int)*2 + failed_high.astype(int)*2 + failed_breakout.astype(int)*2 + low_break.astype(int)*3 + ma_bear.astype(int)*2 + ((c<c.shift(1))&(d.volume_ratio>=1.3)).astype(int))
    # Hold failed-high / false-breakout warnings briefly; don't instantly forget them.
    risk_recent=d.risk_score.rolling(5,min_periods=1).max()
    prior_box_high=d.High.shift(1).rolling(40,min_periods=30).max()
    prior_box_low=d.Low.shift(1).rolling(40,min_periods=30).min()
    box_width=prior_box_high/prior_box_low-1
    flat20=(d.MA20/d.MA20.shift(10)-1).abs()<.035
    flat60=(d.MA60/d.MA60.shift(10)-1).abs()<.045
    box=(box_width<.22)&flat20&flat60
    up=(d.MA20>d.MA60)&(d.MA20>d.MA20.shift(5)*1.004)&(c>d.MA60)&(~low_break)
    down=((d.MA20<d.MA60)&slope_down&ma_weak)|(low_break&ma_weak)
    warning=((risk_recent>=4)&(~box)) | (failed_breakout&ma_weak)
    # priority: confirmed down / transition / box / up / neutral
    regime=np.select([down,warning,box,up],['하락','하락 전환 의심','횡보','상승'],default='중립')
    d['regime']=pd.Series(regime,index=d.index)
    d['box_low']=prior_box_low
    d['box_high']=prior_box_high
    d['buy_blocked']=d.regime.isin(['하락','하락 전환 의심','중립'])
    # Uptrend: require actual pullback to rising MA or established support, then reclaim.
    pullback_test=(d.Low.shift(1).rolling(5,min_periods=3).min()<=d.MA20.shift(1).rolling(5,min_periods=3).max()*1.012)
    higher_low=(d.Low.shift(1).rolling(5,min_periods=3).min()>d.Low.shift(7).rolling(12,min_periods=8).min()*.985)
    confirmation=(c>d.High.shift(1))&(c>d.Open)&(c>c.shift(2))
    up_buy=(d.regime=='상승')&pullback_test&higher_low&confirmation&(c>d.MA20)&(c<d.MA20*1.065)
    # Box: bottom 20% of *previous* box, with BB lower contact and a rejection.
    box_pos=(c-prior_box_low)/(prior_box_high-prior_box_low).replace(0,np.nan)
    near_box=(d.Low<=prior_box_low+(prior_box_high-prior_box_low)*.18)&(c>=prior_box_low*.99)
    bb_test=d.Low<=d.BB_lower+atr*.45
    box_buy=(d.regime=='횡보')&near_box&bb_test&d.reversal_candle&(box_pos<.35)
    # Downtrend reversal is separate: a confirmed breakout, followed by a later retest.
    # The 'transition' regime does not buy on a one-day bounce.
    breakout_recent=bars_since_breakout.between(2,12)
    retest=(d.Low<=breakout_level*1.025)&(d.Low>=breakout_level*.965)&(c>breakout_level)&confirmation
    recovered=(d.MA20>=d.MA20.shift(5))&(c>d.MA20)&(c>d.MA60*.985)
    reversal_buy=(d.regime.isin(['상승','횡보']))&breakout_recent&retest&recovered&(~failed_breakout)
    # v4.7.3: Separate a WATCH zone from a confirmed, tradable reversal.
    # Support zones use only prior bars or indicators computed at the current close.
    # BB contact by itself is never sufficient for an executed buy.
    prior_floor=d.Low.shift(1).rolling(100,min_periods=35).min()
    prior_floor60=d.Low.shift(1).rolling(60,min_periods=25).min()
    support_tolerance=pd.concat([atr*.75,c*.018],axis=1).max(axis=1)
    bb_contact=(d.Low<=d.BB_lower+atr*.55)
    prior_floor_contact=(d.Low<=prior_floor+support_tolerance)|(d.Low<=prior_floor60+support_tolerance)
    pivot_contact=near(recent_low,support_tolerance)
    trend_contact=near(d.trend_support,support_tolerance)&(d.trend_support>0)
    fib_contact=pd.Series(False,index=d.index)
    for f in [.236,.382,.5,.618]:
        fib_contact=fib_contact|near(d[f'fib_{f}'],atr*.45)
    # 'Confluence' means BB lower + a distinct support source, not many correlated fibs.
    structural_support=prior_floor_contact|pivot_contact|trend_contact
    support_confluence=bb_contact&structural_support
    deep_drop=(c/c.shift(15)<.90)|(c/d.MA20<.94)
    near_recent_bottom=d.Low<=d.Low.shift(1).rolling(15,min_periods=7).min()+atr*.65
    # Early reversal WATCH: explicitly no simulated execution.
    d['bottom_watch']=(bb_contact&deep_drop&near_recent_bottom).fillna(False)
    # Higher-confidence watch when two distinct support concepts overlap.
    d['bottom_priority']=(support_confluence&near_recent_bottom).fillna(False)
    # Confirm after support holds: price recovers above prior close / previous high,
    # with a confirmed reversal candle. Do not buy an unconfirmed falling knife.
    prior_watch=d.bottom_priority.shift(1).rolling(5,min_periods=1).max().fillna(0).astype(bool)
    recent_floor=d.Low.shift(1).rolling(5,min_periods=2).min()
    hold_floor=d.Low>=recent_floor-atr*.65
    bounce_confirm=(d.reversal_candle & (c>d.High.shift(1)) & (c>c.shift(2)))
    # Falling-knife guard: avoid buys in a fresh breakdown or persistent steep decline.
    fresh_break=(c<prior_floor*.975)|(d.Low<recent_floor-atr*1.1)
    steep_decline=(d.MA20<d.MA20.shift(5)*.955)&(c<d.MA20*.93)
    # Exceptional bottom entries need a prior watch, a recovery and no new breakdown.
    bottom_buy=prior_watch&hold_floor&bounce_confirm&(~fresh_break)&(~steep_decline)
    bottom_buy=bottom_buy&(c<=d.MA20*1.025)&(c<=d.BB_lower+atr*2.7)
    # Downtrend entries require TWO support types and stronger evidence.
    bottom_buy=bottom_buy&(~(d.regime=='하락')|support_confluence.rolling(5,min_periods=1).max().astype(bool))
    # Uptrend / box / breakout entries stay independent.
    d['entry_type']=np.select(
        [bottom_buy,up_buy,box_buy,reversal_buy],
        ['바닥 지지 확인','상승 눌림목','박스 하단','돌파 후 재지지'],default='')
    d['buy_candidate']=(
        d.bottom_watch|d.bottom_priority|
        ((d.regime=='상승')&pullback_test)|
        ((d.regime=='횡보')&near_box&bb_test)|
        ((d.regime.isin(['상승','횡보']))&breakout_recent)
    ).fillna(False)
    d['overextended']=(c>d.MA20*1.085)|(c/c.shift(5)>1.13)|(d.RSI14>75)
    # A confirmed bottom exception overrides only the regime block, never price safeguards.
    d['buy_confirmed']=(
        (bottom_buy|(up_buy|box_buy|reversal_buy)&(~d.buy_blocked))&
        (~d.overextended)&(~fresh_break)
    ).fillna(False)
    d['support_hits']=(
        bb_contact.astype(int)+prior_floor_contact.astype(int)+
        pivot_contact.astype(int)+trend_contact.astype(int)+fib_contact.astype(int)
    )
    d['buy_score']=d.support_hits.clip(upper=4)*2+d.reversal_candle.astype(int)*2+(d.volume_ratio>=1.3).astype(int)
    d['pullback']=pullback_test
    d['breakout_retest']=reversal_buy
    d['bottom_watch_reason']=np.select(
        [d.bottom_priority,d.bottom_watch],
        ['BB 하단 + 장기/전저점/추세선 지지 중첩','BB 하단 + 급락 후 저점 접근'],default='')
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
    # Limit displayed confirmations to one per 7 bars; actual simulated trades
    # already enforce a single position and exit cooldown.
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
            pending=('buy',str(r.entry_type),date)
    return pd.DataFrame(events,columns=['signal_date','date','action','reason','price','fraction']),pd.Series([v for _,v in equity],index=[t for t,_ in equity],name='equity',dtype=float),units>0
