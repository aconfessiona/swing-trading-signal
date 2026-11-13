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

# 5.0 C/D: causal structural zones, deduplicated watch events, momentum diagnostics.
VERSION = '5.0-C/D'


def analyze_cd(raw, **settings):
    d = analyze(raw, **settings)
    c = d['Close']
    ema12 = c.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd - signal
    d['MACD_hist'] = hist
    # Dimensionless oscillator: relative to ATR, not arbitrary stock price scale.
    d['momentum_wave'] = (hist / d['ATR'].replace(0, np.nan)).clip(-2, 2)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = up / down.replace(0, np.nan)
    d['RSI14'] = 100 - 100/(1+rs)
    d.loc[(down == 0) & (up > 0), 'RSI14'] = 100
    d.loc[(down == 0) & (up == 0), 'RSI14'] = 50
    d['momentum_state'] = np.where(d.momentum_wave >= 0, '상승 모멘텀', '하락 모멘텀')
    d['momentum_fading'] = np.where(d.momentum_wave >= 0,
        d.momentum_wave < d.momentum_wave.shift(1),
        d.momentum_wave > d.momentum_wave.shift(1))

    # Pivots are recorded on their confirmation bar only. Zone centers are made
    # from historical, already confirmed support/resistance pivots, not future extrema.
    d['support_zone_low'] = np.nan
    d['support_zone_high'] = np.nan
    d['support_strength'] = 0
    d['resistance_zone_low'] = np.nan
    d['resistance_zone_high'] = np.nan
    d['zone_watch'] = False
    d['zone_watch_reason'] = ''
    d['zone_id'] = ''
    d['decision_summary'] = ''
    d['reversal_step'] = ''
    d['support_levels'] = ''
    d['resistance_levels'] = ''
    lows, highs = [], []
    watched = {}  # keyed by stable anchor pivot confirmation date, one event per zone
    cooldown = 10
    for i in range(len(d)):
        r = d.iloc[i]
        date = d.index[i]
        atr = float(r.ATR) if pd.notna(r.ATR) else np.nan
        close = float(r.Close)
        if pd.notna(r.pivot_low_confirmed):
            lows.append((i, float(r.pivot_low_confirmed)))
        if pd.notna(r.pivot_high_confirmed):
            highs.append((i, float(r.pivot_high_confirmed)))
        # Use pivots confirmed within last 250 trading sessions.
        lows = [(j, v) for j, v in lows if i-j <= 250]
        highs = [(j, v) for j, v in highs if i-j <= 250]
        if not np.isfinite(atr) or atr <= 0 or i < 100:
            continue
        tolerance = 0.75 * atr
        def groups(points):
            # Cluster already-confirmed pivots by price; no use of future bars.
            result = []
            for j, v in sorted(points, key=lambda x: x[1]):
                match = next((g for g in result if abs(v - g['center']) <= tolerance), None)
                if match is None:
                    result.append({'center': v, 'members': [(j, v)]})
                else:
                    match['members'].append((j, v))
                    match['center'] = float(np.median([x[1] for x in match['members']]))
            return result
        support_candidates = [g for g in groups(lows) if g['center'] <= close + 0.75*atr]
        resist_candidates = [g for g in groups(highs) if g['center'] >= close - 0.75*atr]
        # Candidate prioritization: close to price, repeated confirmed pivots, recency.
        def rank(g):
            touches = len(g['members'])
            distance = abs(close-g['center']) / atr
            age = i-max(j for j,_ in g['members'])
            return min(touches, 4)*24 + max(0, 25-6*distance) + max(0, 15-age/15)
        sg = max(support_candidates, key=rank) if support_candidates else None
        rg = min(resist_candidates, key=lambda g: (max(0, g['center']-close), -len(g['members']))) if resist_candidates else None
        if sg:
            center = sg['center']
            width = 0.45*atr
            strength = min(100, int(round(rank(sg))))
            d.at[date, 'support_zone_low'] = center-width
            d.at[date, 'support_zone_high'] = center+width
            d.at[date, 'support_strength'] = strength
            d.at[date, 'support_levels'] = f'{center:,.0f} / 확정 저점 {len(sg["members"])}개'
            # One interest event per support anchor; cooldown + price-zone rearm.
            anchor = min(j for j,_ in sg['members'])
            key = f's{anchor}'
            near = float(r.Low) <= center+width and float(r.High) >= center-width
            prev_event = watched.get(key)
            # Only meaningful structural support; BB contact is auxiliary, never enough.
            if near and strength >= 65 and prev_event is None:
                d.at[date, 'zone_watch'] = True
                d.at[date, 'zone_watch_reason'] = f'확정 저점 {len(sg["members"])}개 기반 지지 영역 첫 접근 (매수 아님)'
                d.at[date, 'zone_id'] = key
                watched[key] = i
            elif near and prev_event is not None and i-prev_event >= cooldown:
                # Do not generate repeated markers; persistent zones are represented by a band.
                pass
        if rg:
            center = rg['center']
            d.at[date, 'resistance_zone_low'] = center-0.45*atr
            d.at[date, 'resistance_zone_high'] = center+0.45*atr
            d.at[date, 'resistance_levels'] = f'{center:,.0f} / 확정 고점 {len(rg["members"])}개'
        regime = str(r.regime)
        if regime == 'DOWN':
            step = '하락 추세 · 바닥 탐색'
        elif regime == 'REVERSAL_PREP':
            step = '돌파 후 재지지 확인 대기'
        elif regime == 'REVERSAL_CONFIRMED':
            step = '상승 전환 구조 확인'
        elif regime == 'WEAK':
            step = '상승 구조 훼손 경고'
        elif regime == 'UP':
            step = '상승 추세 유지'
        elif regime == 'RANGE':
            step = '횡보 박스 관찰'
        else:
            step = '구조 확인 대기'
        d.at[date, 'reversal_step'] = step
        if bool(r.buy_blocked):
            decision = f'신규 매수 금지: {r.block_reasons}'
        else:
            decision = '기본 금지 조건 없음 · 진입/손절/목표가 및 손익비 미검증 (매수 신호 아님)'
        if bool(d.at[date, 'zone_watch']):
            decision += ' | 주요 지지 관심 구간'
        d.at[date, 'decision_summary'] = decision
    # Replace noisy A/B watch with structural, deduplicated C/D watch.
    d['watch'] = d['zone_watch'].astype(bool)
    d['watch_reason'] = d['zone_watch_reason']
    return d

# 5.2: proactive partial profit taking; signals known only after daily close.
VERSION = '5.2-C-EARLY-EXIT'

def analyze_51(raw, **settings):
    d = analyze_cd(raw, **settings)
    c=d.Close
    width=(d.BB_upper-d.BB_lower).replace(0,np.nan)
    d['percent_b']=(c-d.BB_lower)/width
    d['bb_touch']=d.High>=d.BB_upper*.99
    d['bb_reentry']=(c.shift(1)>=d.BB_upper.shift(1))&(c<d.BB_upper)
    d['bearish_break']=c<d.Low.shift(1)
    d['hist_fading2']=(d.MACD_hist<d.MACD_hist.shift(1))&(d.MACD_hist.shift(1)<d.MACD_hist.shift(2))
    delta=c.diff();gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean();loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['RSI14']=100-100/(1+gain/loss.replace(0,np.nan))
    d['band_walk']=((d.percent_b>=.8).rolling(5,min_periods=5).sum()>=3)&(d.MA20>d.MA20.shift(5))&(d.regime=='UP')&(d.Close>=d.last_low.fillna(-np.inf)*.99)
    for name,value in [('strong_protect',False),('sell_watch',False),('sell_confirm',False),('sell_reason',''),('sell_score',0),('resistance_near',False),('early_exit',False),('second_exit',False),('short_resistance',np.nan),('mid_resistance',np.nan),('long_resistance',np.nan)]:d[name]=value
    highs=[];protect=False;off_count=0;last_signal=-999;last_level=np.nan;last_event_high=np.nan;last_second=-999
    for i in range(len(d)):
        r=d.iloc[i];idx=d.index[i]
        if pd.notna(r.pivot_high_confirmed):highs.append((i,float(r.pivot_high_confirmed)))
        if i<100 or pd.isna(r.ATR) or r.ATR<=0:continue
        atr=float(r.ATR);price=float(r.Close)
        for window,col in ((60,'short_resistance'),(180,'mid_resistance'),(756,'long_resistance')):
            candidates=[v for j,v in highs if i-j<=window and v>price+.15*atr]
            if candidates:d.at[idx,col]=min(candidates)
        levels=[float(d.at[idx,col]) for col in ('short_resistance','mid_resistance','long_resistance') if pd.notna(d.at[idx,col])]
        near=bool(levels and r.High>=min(levels)-.5*atr)
        d.at[idx,'resistance_near']=near
        touch=bool(r.bb_touch or near)
        if bool(r.band_walk):protect=True;off_count=0
        elif protect:
            off_count=off_count+1 if r.percent_b<.8 and bool(r.hist_fading2) else 0
            broken=pd.notna(r.last_low) and price<r.last_low*.99
            ma_broken=i>=1 and price<r.MA20 and d.Close.iloc[i-1]<d.MA20.iloc[i-1]
            if broken or ma_broken or off_count>=3:protect=False;off_count=0
        d.at[idx,'strong_protect']=protect
        # Mark interest in the explanation only; never clutter chart with repeated circles.
        d.at[idx,'sell_watch']=touch
        score=sum([bool(r.bb_reentry),bool(r.bearish_break),bool(r.hist_fading2),near])
        d.at[idx,'sell_score']=score
        range_like=r.regime in ('RANGE','NEUTRAL') or (pd.notna(r.ADX) and r.ADX<18 and r.regime!='UP')
        # Proactive exit uses close-known high touch + RSI or structure confluence.
        # In a strong trend, require both structural resistance and momentum fading.
        early=touch and (bool(pd.notna(r.RSI14) and r.RSI14>=65) or near)
        if protect:early=bool(near and r.hist_fading2 and pd.notna(r.RSI14) and r.RSI14>=70)
        elif not range_like:early=bool(early and (near or r.hist_fading2))
        # New wave requires fresh extension > 1 ATR and at least 12 bars since signal.
        fresh=i-last_signal>=12 and (not np.isfinite(last_event_high) or r.High>last_event_high+atr)
        early=bool(early and fresh)
        if early:
            d.at[idx,'early_exit']=True
            d.at[idx,'sell_confirm']=True
            d.at[idx,'sell_reason']='선제적 1차 익절 후보: 상단/저항 + 과열 또는 저항 중첩'
            last_signal=i;last_event_high=float(r.High);last_level=price
        else:
            # Second-stage confirmation requires a NEW high after a prior first-stage event.
            second=bool(np.isfinite(last_level) and i-last_signal>=3 and i-last_second>=12 and r.High>last_event_high+atr and (r.bb_reentry or r.bearish_break) and (r.hist_fading2 or near))
            if second:
                d.at[idx,'second_exit']=True;d.at[idx,'sell_confirm']=True
                d.at[idx,'sell_reason']='새 고점 확장 이후 반락: 2차 익절 후보'
                last_second=i;last_event_high=float(r.High)
            elif protect:d.at[idx,'sell_reason']='밴드 워크 보호: 상단 접촉만으로 매도하지 않음'
            elif touch:d.at[idx,'sell_reason']='상단/저항 관찰: 표시 생략'
    return d

def simulate_held_position(d, entry_date, fee_rate=.00015, slippage=.0005):
    """Illustrative long-only position entered on a user-selected date's NEXT open.
    Signal on close -> execution next open; a stop hit intraday executes at stop
    or worse next-open gap. Does not imply that the entry was a valid buy signal.
    """
    entry_loc=d.index.searchsorted(pd.Timestamp(entry_date),side='right')
    if entry_loc>=len(d):raise ValueError('선택 날짜 이후 거래일이 없습니다.')
    entry=float(d.Open.iloc[entry_loc])*(1+slippage)
    qty=1.;sold=0.;first_high=entry;initial_qty=1.
    stop=entry-2*float(d.ATR.iloc[entry_loc])
    if not np.isfinite(stop) or stop<=0:raise ValueError('진입일 ATR 계산이 불가능합니다.')
    events=[{'date':d.index[entry_loc],'action':'가상 진입','price':entry,'fraction':0.,'reason':'사용자 선택일 다음 시가 · 매수 신호 아님'}]
    pending=None; first_sell_level=np.nan;last_sale_i=-999;realized=0.
    for i in range(entry_loc+1,len(d)):
        r=d.iloc[i]; idx=d.index[i]
        op=float(r.Open); low=float(r.Low)
        # Fill prior-close signal at next open. Protective gap exit takes priority.
        if op<=stop or low<=stop:
            price=min(op,stop)*(1-slippage)
            realized+=qty*(price-entry)-qty*price*fee_rate
            events.append({'date':idx,'action':'보호 청산','price':price,'fraction':qty,'reason':'손절/추적 손절선 도달'})
            qty=0;break
        if pending and qty>0:
            fraction=min(pending['fraction'],qty)
            price=op*(1-slippage)
            realized+=fraction*(price-entry)-fraction*price*fee_rate
            qty-=fraction;sold+=fraction;last_sale_i=i
            events.append({'date':idx,'action':pending['action'],'price':price,'fraction':fraction,'reason':pending['reason']})
            if sold>=.3 and sold<.6:first_sell_level=price
            pending=None
        if qty<=1e-8:break
        # Stop may only tighten. Today's close-derived stop is effective tomorrow.
        if i>=entry_loc+2:
            prior=d.iloc[i-1]
            if pd.notna(prior.last_low) and pd.notna(prior.ATR):
                stop=max(stop,float(prior.last_low)-.5*float(prior.ATR))
        # Risk-off conditions at today's close -> next-open exit.
        if (pd.notna(r.last_low) and r.Close<r.last_low*.99) or (i>=2 and all(d.Close.iloc[j]<d.MA20.iloc[j] for j in range(i-2,i+1)) and r.regime in ('DOWN','WEAK')):
            pending={'action':'보호 청산','fraction':qty,'reason':'종가 기준 구조 훼손'}
        elif sold<.3 and bool(r.early_exit) and i>last_sale_i:
            pending={'action':'1차 매도 30%','fraction':.3,'reason':str(r.sell_reason)}
        elif .3-1e-8<=sold<.6 and i>last_sale_i+1 and bool(r.second_exit):
            # Require price expansion beyond first exit and a new resistance event.
            if r.High>first_sell_level+float(r.ATR):
                pending={'action':'2차 매도 30%','fraction':.3,'reason':'새 가격 확장 + 저항 반락'}
    # If a signal is pending on the last date, don't fabricate a fill.
    unrealized=qty*(float(d.Close.iloc[-1])-entry)
    return events, {'entry':entry,'remaining':qty,'realized_pnl_per_initial_share':realized-entry*fee_rate,'unrealized_pnl_per_initial_share':unrealized,'pending':pending}

# 5.3: separate diagnostic sell opportunities from position executions.
VERSION = '5.3.1-C-CLEAN-CHART'

def analyze_53(raw, **settings):
    d = analyze_51(raw, **settings)
    d['early_exit'] = False
    d['second_exit'] = False
    d['sell_confirm'] = False
    d['sell_candidate_type'] = ''
    d['sell_reason'] = ''
    d['sell_candidate_score'] = 0
    d['sell_block_reason'] = ''
    # State per upward swing: no repeated signal while price oscillates in one zone.
    last_event_i = -1000
    last_event_high = np.nan
    last_event_low = np.nan
    last_event_type = ''
    for i in range(100, len(d)):
        r = d.iloc[i]
        idx = d.index[i]
        atr = float(r.ATR) if pd.notna(r.ATR) else np.nan
        if not np.isfinite(atr) or atr <= 0 or pd.isna(r.BB_upper):
            continue
        bb_upper = float(r.BB_upper)
        touch = float(r.High) >= bb_upper * .985
        # Only resistances that were known at the close, including the current bar.
        levels = [float(r[k]) for k in ('short_resistance','mid_resistance','long_resistance') if pd.notna(r[k])]
        near = bool(levels and any(abs(float(r.High)-v) <= .75*atr for v in levels))
        rsi = float(r.RSI14) if pd.notna(r.RSI14) else 50.
        fading = bool(r.hist_fading2)
        reentry = bool(r.bb_reentry)
        reversal = bool(r.bearish_break or reentry)
        range_like = str(r.regime) in ('RANGE','NEUTRAL') or (pd.notna(r.ADX) and float(r.ADX)<18 and str(r.regime)!='UP')
        strong = bool(r.strong_protect)
        # This is a candidate signal, not a fill or prediction of the top.
        if range_like:
            score = int(touch)*2 + int(near) + int(rsi>=60) + int(fading) + int(reversal)
            proactive = touch and (rsi>=60 or near) and score>=3
        elif strong:
            score = int(touch) + int(near)*2 + int(rsi>=75) + int(fading) + int(reversal)
            proactive = touch and near and rsi>=75 and fading and score>=5
        else:
            score = int(touch)*2 + int(near) + int(rsi>=65) + int(fading) + int(reversal)
            proactive = touch and (rsi>=65 or near) and (fading or near) and score>=4
        reactive = (touch or near or (i>0 and bool(d.sell_watch.iloc[i-1]))) and reversal and (fading or near)
        d.at[idx,'sell_candidate_score'] = score
        # A new wave requires a reset below MA20 or a fresh 1.2 ATR extension.
        new_wave = (np.isfinite(last_event_high) and float(r.High)>last_event_high+1.2*atr and i-last_event_i>=5)
        reset = (np.isfinite(last_event_low) and i-last_event_i>=8 and
                 d.Close.iloc[max(0,last_event_i+1):i].lt(d.MA20.iloc[max(0,last_event_i+1):i]).sum()>=3)
        rearmed = last_event_i<0 or ((i-last_event_i)>=10 and (new_wave or reset))
        if not rearmed:
            d.at[idx,'sell_block_reason']='동일 상승 파동 중복 후보 제한'
            continue
        if proactive:
            d.at[idx,'early_exit']=True
            d.at[idx,'sell_confirm']=True
            d.at[idx,'sell_candidate_type']='선제 익절 후보'
            d.at[idx,'sell_reason']=f'볼린저 상단 접근 · RSI {rsi:.1f} · 저항 중첩 {"예" if near else "아니오"} · 모멘텀 둔화 {"예" if fading else "아니오"}'
        elif reactive and not strong:
            d.at[idx,'second_exit']=True
            d.at[idx,'sell_confirm']=True
            d.at[idx,'sell_candidate_type']='반락 익절 후보'
            d.at[idx,'sell_reason']='볼린저 상단/저항 접근 후 가격 반락 및 모멘텀 둔화'
        elif strong:
            d.at[idx,'sell_block_reason']='밴드 워크 보호'
        else:
            d.at[idx,'sell_block_reason']='과열·저항·반락 조건 부족'
        if bool(d.at[idx,'sell_confirm']):
            last_event_i=i
            last_event_high=float(r.High)
            last_event_low=float(r.Low)
            last_event_type=str(d.at[idx,'sell_candidate_type'])
    return d


def simulate_held_position_53(d, entry_date, fee_rate=.00015, slippage=.0005):
    """Next-open entry and fills, initial-share 30/30/40; protective stop active from prior close."""
    entry_loc=d.index.searchsorted(pd.Timestamp(entry_date),side='right')
    if entry_loc>=len(d):raise ValueError('선택 날짜 이후 거래일이 없습니다.')
    entry=float(d.Open.iloc[entry_loc])*(1+slippage)
    atr=float(d.ATR.iloc[entry_loc])
    if not np.isfinite(atr) or atr<=0:raise ValueError('진입일 ATR 계산 불가')
    stop=entry-2*atr
    qty=1.;sold=0.;realized=-entry*fee_rate
    pending=None;first_exit_high=np.nan;first_exit_i=-1
    events=[{'date':d.index[entry_loc],'action':'가상 진입','price':entry,'fraction':0.,'reason':'선택일 다음 시가 · 매수 신호 아님'}]
    for i in range(entry_loc+1,len(d)):
        r=d.iloc[i];idx=d.index[i];op=float(r.Open);low=float(r.Low)
        # Stop is based only on the previous session's information.
        if op<=stop or low<=stop:
            price=min(op,stop)*(1-slippage)
            realized+=qty*(price-entry)-qty*price*fee_rate
            events.append({'date':idx,'action':'보호 청산','price':price,'fraction':qty,'reason':'기존 손절/추적 손절선 도달'})
            qty=0.;pending=None;break
        if pending is not None:
            fraction=min(qty,pending['fraction']);price=op*(1-slippage)
            realized+=fraction*(price-entry)-fraction*price*fee_rate
            qty-=fraction;sold+=fraction
            events.append({'date':idx,'action':pending['action'],'price':price,'fraction':fraction,'reason':pending['reason']})
            if pending['action']=='1차 매도 30%':
                first_exit_high=pending['signal_high'];first_exit_i=i
            pending=None
        if qty<=1e-8:break
        # Effective next trading session; never backdate today's stop.
        if pd.notna(r.last_low) and pd.notna(r.ATR):
            stop=max(stop,float(r.last_low)-.5*float(r.ATR))
        risk_off=(pd.notna(r.last_low) and r.Close<r.last_low*.99) or (i>=2 and all(d.Close.iloc[j]<d.MA20.iloc[j] for j in range(i-2,i+1)) and r.regime in ('DOWN','WEAK'))
        if risk_off:
            pending={'action':'보호 청산','fraction':qty,'reason':'종가 기준 구조 훼손','signal_high':float(r.High)}
        elif sold<.3-1e-8 and bool(r.sell_confirm):
            pending={'action':'1차 매도 30%','fraction':.3,'reason':str(r.sell_reason),'signal_high':float(r.High)}
        elif .3-1e-8<=sold<.6-1e-8 and bool(r.sell_confirm) and i>first_exit_i+2 and np.isfinite(first_exit_high) and r.High>first_exit_high+float(r.ATR):
            pending={'action':'2차 매도 30%','fraction':.3,'reason':'새 상승 확장 후 별도 매도 후보','signal_high':float(r.High)}
    unrealized=qty*(float(d.Close.iloc[-1])-entry)
    return events,{'entry':entry,'remaining':qty,'realized_pnl_per_initial_share':realized,'unrealized_pnl_per_initial_share':unrealized,'pending':pending}
