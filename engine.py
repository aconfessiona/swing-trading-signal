"""Swing 5.0 A/B: causal regime and buy-block diagnostics, NOT trade signals."""
import numpy as np
import pandas as pd

VERSION = '5.0-A/B'
REQUIRED = ('Open', 'High', 'Low', 'Close', 'Volume')


def _adx(d, n=14):
    h,l,c=d.High,d.Low,d.Close
    up=h.diff(); down=-l.diff()
    plus=up.where((up>down)&(up>0),0.0)
    minus=down.where((down>up)&(down>0),0.0)
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    p=100*plus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr.replace(0,np.nan)
    m=100*minus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr.replace(0,np.nan)
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    adx=dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return atr,adx,p,m


def prepare(raw):
    d=raw.copy()
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    d.columns=[str(x).strip().capitalize() for x in d.columns]
    missing=[x for x in REQUIRED if x not in d.columns]
    if missing:raise ValueError('누락된 컬럼: '+', '.join(missing))
    d.index=pd.to_datetime(d.index)
    if d.index.tz is not None:d.index=d.index.tz_localize(None)
    d=d.loc[~d.index.duplicated(keep='last')].sort_index()
    for c in REQUIRED:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=REQUIRED)
    d=d[(d.High>=d.Low)&(d.High>=d[['Open','Close']].max(axis=1))&(d.Low<=d[['Open','Close']].min(axis=1))&(d.Close>0)]
    if len(d)<140:raise ValueError('국면 판별에는 최소 140개 일봉이 필요합니다.')
    c=d.Close
    d['MA20']=c.rolling(20).mean();d['MA60']=c.rolling(60).mean()
    std=c.rolling(20).std(ddof=0)
    d['BB_upper']=d.MA20+2*std;d['BB_lower']=d.MA20-2*std
    d['ATR'],d['ADX'],d['PLUS_DI'],d['MINUS_DI']=_adx(d)
    d['MA20_slope5']=d.MA20/d.MA20.shift(5)-1
    d['MA60_slope10']=d.MA60/d.MA60.shift(10)-1
    d['vol_ratio']=d.Volume/d.Volume.shift(1).rolling(20).mean().replace(0,np.nan)
    # A pivot at t is confirmed only at t+2. Never expose it earlier.
    h=d.High; l=d.Low
    hp=(h>h.shift(1))&(h>h.shift(2))&(h>h.shift(-1))&(h>h.shift(-2))
    lp=(l<l.shift(1))&(l<l.shift(2))&(l<l.shift(-1))&(l<l.shift(-2))
    d['pivot_high_confirmed']=h.where(hp).shift(2)
    d['pivot_low_confirmed']=l.where(lp).shift(2)
    # Last confirmed pivot is available on the confirmation day; prior pivot is saved separately.
    d['last_high']=d.pivot_high_confirmed.ffill()
    d['last_low']=d.pivot_low_confirmed.ffill()
    d['prev_high']=np.nan;d['prev_low']=np.nan
    last_hi=prev_hi=np.nan;last_lo=prev_lo=np.nan
    for i in range(len(d)):
        hi=d.pivot_high_confirmed.iloc[i];lo=d.pivot_low_confirmed.iloc[i]
        if pd.notna(hi):prev_hi,last_hi=last_hi,hi
        if pd.notna(lo):prev_lo,last_lo=last_lo,lo
        d.iat[i,d.columns.get_loc('prev_high')]=prev_hi
        d.iat[i,d.columns.get_loc('prev_low')]=prev_lo
    d['prior_40_high']=d.High.shift(1).rolling(40).max()
    d['prior_40_low']=d.Low.shift(1).rolling(40).min()
    d['prior_40_mid']=(d.prior_40_high+d.prior_40_low)/2
    d['range_width']=(d.prior_40_high-d.prior_40_low)/d.prior_40_mid
    d['atr_median60']=d.ATR.shift(1).rolling(60).median()
    return d


def analyze(raw, ma_below_days=3, breakout_pct=0.01, retest_days=10, min_rr=1.8, adx_ref=18):
    """Return one row per bar, all states computed using information available at that close.
    This is a diagnostic engine. It NEVER places orders or claims a buy confirmation.
    """
    d=prepare(raw)
    for col,default in [('regime','WARMUP'),('phase',''),('buy_blocked',True),('block_reasons',''),('watch',False),('watch_reason',''),('transition_reason',''),('reversal_route','')]:d[col]=default
    below=0;regime='WARMUP';phase='';breakout_level=np.nan;breakout_age=0;last_broken_low=np.nan
    # Conservative state-machine; structural invalidation overrides positive MA signals.
    for i in range(len(d)):
        r=d.iloc[i];c=float(r.Close);a=float(r.ATR) if pd.notna(r.ATR) else np.nan
        if pd.notna(r.MA20) and c<r.MA20:below+=1
        else:below=0
        if i<100:continue
        prior=d.iloc[i-1]
        last_low=float(prior.last_low) if pd.notna(prior.last_low) else np.nan
        last_high=float(prior.last_high) if pd.notna(prior.last_high) else np.nan
        # A newly confirmed pivot is available today, but cannot be retroactively used.
        hl=bool(pd.notna(r.prev_low) and pd.notna(r.last_low) and pd.notna(a) and r.last_low-r.prev_low>=.3*a)
        ll=bool(pd.notna(r.prev_low) and pd.notna(r.last_low) and r.last_low<r.prev_low)
        lh=bool(pd.notna(r.prev_high) and pd.notna(r.last_high) and r.last_high<r.prev_high)
        hh=bool(pd.notna(r.prev_high) and pd.notna(r.last_high) and r.last_high>r.prev_high)
        broken_low=bool(pd.notna(last_low) and c<last_low*.99)
        bear_structure=bool(lh and ll and r.MA20_slope5<0)
        trend_down=bool(r.MA20_slope5<0 and r.MA60_slope10<0 and c<r.MA60)
        bearish=broken_low or bear_structure or (trend_down and below>=ma_below_days and r.MINUS_DI>r.PLUS_DI)
        weak=below>=ma_below_days or (pd.notna(last_high) and prior.Close>last_high*1.01 and c<last_high)
        uptrend=bool(hh and hl and c>r.MA60 and r.MA20_slope5>0 and r.MA60_slope10>0)
        sideways=bool(pd.notna(r.range_width) and .12<=r.range_width<=.30 and abs(r.MA60_slope10)<=.02 and r.ADX<20)
        reason=''
        # On confirmed bear, reset any incomplete reversal setup.
        if bearish:
            if regime!='DOWN':reason='주요 저점 이탈 또는 하락 가격구조 확정'
            regime='DOWN';phase='';breakout_level=np.nan;breakout_age=0
            if broken_low:last_broken_low=last_low
        elif regime=='DOWN':
            # A breakout must be relative to a prior confirmed swing high, not today's newly confirmed high.
            if pd.notna(last_high) and c>=last_high*(1+breakout_pct) and c>r.MA20:
                regime='REVERSAL_PREP';phase='RETEST';breakout_level=last_high;breakout_age=0
                reason='확정 전고점 돌파: 재지지 대기'
            elif hl and pd.notna(last_high) and c>=last_high*(1+breakout_pct) and c>r.MA20:
                regime='REVERSAL_PREP';phase='HL';reason='저점 상승 및 구조 돌파'
        elif regime=='REVERSAL_PREP':
            breakout_age+=1
            if breakout_age>retest_days or (pd.notna(breakout_level) and c<breakout_level*.99):
                regime='DOWN';phase='';reason='재지지 기한 초과 또는 돌파 가격 붕괴'
            elif phase=='RETEST' and pd.notna(breakout_level):
                # Requires a retest after breakout day, followed by a separate confirmation day.
                if breakout_age>=1 and r.Low<=breakout_level+0.5*a and c>=breakout_level*.99:
                    phase='RETEST_SEEN';reason='재지지 관찰: 반등 확인 대기'
            elif phase=='RETEST_SEEN' and c>prior.High and c>=breakout_level and c>r.MA20:
                regime='REVERSAL_CONFIRMED';phase='';reason='재지지 후 전일 고가 종가 돌파'
        elif regime=='REVERSAL_CONFIRMED':
            if weak:regime='WEAK';reason='20일선 이탈로 재검토'
            elif uptrend:regime='UP';reason='HH·HL 상승 구조 정착'
        else:
            if regime=='UP' and weak:regime='WEAK';reason='상승 추세 훼손 경고'
            elif uptrend:regime='UP';reason='상승 가격 구조 확인'
            elif sideways:regime='RANGE';reason='박스 및 낮은 추세 강도'
            elif regime=='WEAK':
                if uptrend and not weak:regime='UP';reason='상승 구조 회복'
                elif sideways:regime='RANGE';reason='횡보 구조 확정'
            else:
                # Unknown/transition is blocked, never automatically labeled an uptrend.
                regime='NEUTRAL'
        watch_reasons=[]
        if pd.notna(r.BB_lower) and r.Low<=r.BB_lower+0.35*a:watch_reasons.append('볼린저 하단')
        if pd.notna(r.prior_40_low) and abs(r.Low-r.prior_40_low)<=.6*a:watch_reasons.append('40일 과거 지지')
        if pd.notna(last_low) and abs(r.Low-last_low)<=.6*a:watch_reasons.append('확정 스윙 저점')
        blocks=[]
        if regime in ('DOWN','REVERSAL_PREP','WEAK','NEUTRAL','WARMUP'):blocks.append('국면 차단: '+regime)
        if broken_low:blocks.append('확정 지지선 1% 이탈')
        if c>r.MA20*1.08:blocks.append('20일선 대비 8% 과열')
        if i>=5 and c/d.Close.iloc[i-5]>1.13:blocks.append('5거래일 13% 초과 상승')
        if pd.notna(last_high) and last_high>c and last_high/c-1<.03:blocks.append('저항까지 3% 미만')
        if pd.notna(r.atr_median60) and a>2*r.atr_median60:blocks.append('ATR 급증')
        # RR is deliberately NOT claimed here: an actual entry, stop and target setup is required.
        d.at[d.index[i],'regime']=regime
        d.at[d.index[i],'phase']=phase
        d.at[d.index[i],'buy_blocked']=bool(blocks)
        d.at[d.index[i],'block_reasons']=' | '.join(blocks) if blocks else '기본 금지 조건 없음 (진입 확정 아님)'
        d.at[d.index[i],'watch']=bool(watch_reasons)
        d.at[d.index[i],'watch_reason']=' + '.join(watch_reasons)
        d.at[d.index[i],'transition_reason']=reason
        d.at[d.index[i],'reversal_route']='전고점 돌파 후 재지지' if regime in ('REVERSAL_PREP','REVERSAL_CONFIRMED') else ''
    d.attrs['settings']={'ma_below_days':ma_below_days,'breakout_pct':breakout_pct,'retest_days':retest_days,'min_rr':min_rr,'adx_ref':adx_ref}
    return d
