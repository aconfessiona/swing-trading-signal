"""Swing 4.4 mobile research dashboard; no brokerage integration."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
# 핵심 분석 함수는 이 파일에 내장되어 배포 버전 불일치를 방지합니다.

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
    d['entry_type']=np.select([up_buy,box_buy,reversal_buy],['상승 눌림목','박스 하단','돌파 후 재지지'],default='')
    d['buy_candidate']=((d.regime=='상승')&pullback_test)|((d.regime=='횡보')&near_box&bb_test)|((d.regime.isin(['상승','횡보']))&breakout_recent)
    d['overextended']=(c>d.MA20*1.085)|(c/c.shift(5)>1.13)|(d.RSI14>75)
    d['buy_confirmed']=(up_buy|box_buy|reversal_buy)&(~d.overextended)&(~d.buy_blocked)
    d['support_hits']=(near(d.BB_lower,atr*.45).astype(int)+near(recent_low,atr*.6).astype(int)+near(d.trend_support,atr*.5).astype(int))
    for f in [.236,.382,.5,.618]:d['support_hits']+=near(d[f'fib_{f}'],atr*.4).astype(int)
    d['buy_score']=d.support_hits.clip(upper=4)*2+d.reversal_candle.astype(int)*2+(d.volume_ratio>=1.3).astype(int)
    d['pullback']=pullback_test
    d['breakout_retest']=reversal_buy
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
            pending=('buy',str(r.entry_type),date)
    return pd.DataFrame(events,columns=['signal_date','date','action','reason','price','fraction']),pd.Series([v for _,v in equity],index=[t for t,_ in equity],name='equity',dtype=float),units>0


st.set_page_config(page_title='스윙 4.7',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding:0.7rem 0.6rem 2rem;max-width:1200px}h1{font-size:1.65rem!important}h2,h3{font-size:1.2rem!important}div[data-testid="stPlotlyChart"]{overflow:hidden}@media(max-width:600px){.stRadio>div{gap:.3rem}}</style>''',unsafe_allow_html=True)
st.title('📈 스윙 매매 4.7.2')
st.caption('가격·지지저항 중심 | 거래량 가점 | 30/30/40 분할 매도 | 연구용')
with st.expander('⚙️ 종목 및 백테스트 설정',expanded=True):
    ticker=st.text_input('종목코드',value='373220.KS').strip().upper()
    a,b=st.columns(2)
    years=a.selectbox('데이터 기간',[2,3,5],index=1)
    view=b.selectbox('차트 표시',['3개월','6개월','1년','전체'],index=1)
    c,e=st.columns(2)
    fee=c.number_input('편도 비용 (bp)',min_value=0,max_value=200,value=15,step=5)
    hold=e.slider('보유기간 참고값 (자동 청산 없음)',10,80,40)
    atr=st.slider('ATR 보호 청산 배수',1.5,4.5,2.5,.25)
@st.cache_data(ttl=1800,show_spinner=False)
def load(symbol,years):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    start=end-dt.timedelta(days=366*years)
    raw=yf.download(symbol,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=False)
    if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
    if raw.empty:raise ValueError('가격 데이터가 없습니다. 종목코드를 확인하세요.')
    return raw
uploaded=st.file_uploader('선택: 일봉 CSV (Date,Open,High,Low,Close,Volume)',type=['csv'])
try:
    if uploaded:
        raw=pd.read_csv(uploaded)
        raw.columns=[str(c).strip().capitalize() for c in raw.columns]
        if 'Date' not in raw:raise ValueError('CSV에는 Date 열이 필요합니다.')
        raw=raw.set_index('Date')
    else:raw=load(ticker,years)
    d=prepare(raw)
    if len(d)<90:raise ValueError('최소 90개 일봉이 필요합니다.')
    ev,eq,opened=simulate(d,fee,atr,hold)
except Exception as exc:
    st.error(f'데이터 처리 오류: {exc}');st.stop()
# Version mismatch guard: an older engine.py can silently remain deployed.
required={'regime','buy_blocked','buy_candidate','buy_confirmed','entry_type','risk_score'}
missing=required-set(d.columns)
if missing:
    st.error('국면 지표 생성 오류: '+', '.join(sorted(missing))+' — 입력 데이터와 분석 함수를 점검해 주세요.')
    st.stop()
last=d.iloc[-1]
regime=str(last['regime'])
blocked='예' if bool(last['buy_blocked']) else '아니오'
candidate='예' if bool(last['buy_candidate']) else '아니오'
confirmed='예' if bool(last['buy_confirmed']) else '아니오'
last_close=float(last['Close'])
st.info(f'현재 국면 {regime} | 매수 제한 {blocked} | 최근 일봉: {d.index[-1]:%Y-%m-%d} | 종가 {last_close:,.0f} | 매수 후보 {candidate} | 매수 확정 {confirmed}')
st.caption('국면 판별 → 매수 금지 필터 → 국면별 매수. 상승 눌림목 / 박스 하단 / 돌파 후 재지지로 분리합니다. 당일 종가 기준이며 장중 미확정 데이터는 제외하세요.')
with st.expander('🔍 최근 국면과 매수 차단 이유',expanded=False):
    cols=['Close','regime','risk_score','ma20_below_days','failed_high','failed_breakout','buy_blocked','entry_type','buy_confirmed']
    st.dataframe(d[cols].tail(25).iloc[::-1],use_container_width=True)
show=st.radio('차트 표시',['핵심만','지지·저항 추가','모든 지표'],horizontal=True)
markers=st.checkbox('후보 신호 표시 (분석용)',value=False)
all_trades=st.checkbox('모든 체결 마커 표시',value=False,help='기본은 선택한 체결만 차트에 표시합니다.')
visible_ev=ev[ev.action!='잔여 청산'].copy() if not ev.empty else ev.copy()
trade_ev=visible_ev[visible_ev.date.isin(d.tail({'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[view]).index)].copy() if not visible_ev.empty else pd.DataFrame()
# 선택 메뉴에는 기간 내 전체 거래를 제공하되, 기본 차트는 체결 마커를 표시하지 않습니다.
options=['선택 안 함']
if not trade_ev.empty:
    options += [f"{pd.Timestamp(r.date):%Y-%m-%d} · {r.action} · {float(r.price):,.0f}원" for _,r in trade_ev.iloc[::-1].iterrows()]
selected=st.selectbox('강조할 모의 체결 (선택한 봉만 표시)',options,index=1 if len(options)>1 else 0)
n={'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[view]
p=d.tail(n);x=p.index.strftime('%Y-%m-%d').tolist()
fig=go.Figure()
fig.add_trace(go.Candlestick(x=x,open=p.Open,high=p.High,low=p.Low,close=p.Close,name='일봉',increasing_line_color='#d64b4b',decreasing_line_color='#2778c4',increasing_fillcolor='#d64b4b',decreasing_fillcolor='#2778c4'))
def line(col,name,color,width=1.5,dash='solid'):
    fig.add_trace(go.Scatter(x=x,y=p[col],mode='lines',name=name,line=dict(color=color,width=width,dash=dash),hoverinfo='skip'))
line('BB_upper','BB 상단','#8493b7');line('BB_lower','BB 하단','#8493b7')
line('MA20','20일선','#9150b1',2.3);line('MA60','60일선','#32679d',2.3)
if show!='핵심만':
    line('support','확정 전저점','#129c77',1.5,'dot')
    line('resistance','확정 전고점','#d08a38',1.5,'dot')
    line('trend_support','하단 추세선','#13a07c',1.5,'dash')
if show=='모든 지표':
    for f in [.236,.382,.5,.618]:line(f'fib_{f}',f'Fib {f}','#a7a7a7',1,'dot')
    line('weekly_prior_high','주봉 전고점','#bd6c48',1.3,'dash')
    line('weekly_bb_upper','주봉 BB 상단','#ad9244',1.3,'dot')
def mark(dates,values,name,color,symbol,size=11,outline=False,custom=None):
    if len(dates)==0:return
    xx=pd.DatetimeIndex(pd.to_datetime(dates)).strftime('%Y-%m-%d').tolist()
    kw=dict(x=xx,y=values,mode='markers',name=name,
        marker=dict(symbol=symbol,size=size,color='rgba(255,255,255,0.15)' if outline else color,
                    line=dict(color=color,width=2 if outline else 1)))
    if custom is not None:
        kw['customdata']=custom
        kw['hovertemplate']='%{x}<br>'+name+'<br>가격: %{y:,.0f}<br>%{customdata}<extra></extra>'
    else:
        kw['hovertemplate']='%{x}<br>'+name+'<br>가격: %{y:,.0f}<extra></extra>'
    fig.add_trace(go.Scatter(**kw))

if markers:
    buy=p[p.buy_candidate];sell=p[p.sell_candidate]
    mark(buy.index,buy.Close,'매수 관심','#13a074','circle-open',8,True)
    mark(sell.index,sell.Close,'매도 관심','#d99a2b','circle-open',8,True)
if markers:
    confirmed=p[p.buy_confirmed]
    mark(confirmed.index,confirmed.Close,'매수 확정 (종가)','#13995c','triangle-up-open',9,True)

# 차트에는 가격 라벨을 상시 표시하지 않음: 봉과 마커가 가려지는 현상 방지.
# 선택한 거래에만 세로선 및 화살표를 표시하고 상세 정보는 차트 밖에 출력.
if not trade_ev.empty:
    event_specs=[
        ('매수','#13995c','triangle-up'),
        ('1차 30% 매도','#dfaa25','circle'),
        ('2차 30% 매도','#ed7e2b','diamond'),
    ]
    for action,color,symbol in event_specs:
        e=trade_ev[trade_ev.action==action] if all_trades else trade_ev.iloc[0:0]
        if e.empty:continue
        details=['신호일: '+pd.Timestamp(t).strftime('%Y-%m-%d') for t in e.signal_date]
        mark(e.date,e.price.astype(float).tolist(),'모의 '+action,color,symbol,11,custom=details)
    if selected!='선택 안 함':
        selected_date=selected.split(' · ')[0]
        selected_action=selected.split(' · ')[1]
        hit=trade_ev[(trade_ev.date.dt.strftime('%Y-%m-%d')==selected_date)&(trade_ev.action==selected_action)]
        if not hit.empty:
            r=hit.iloc[0]
            date=pd.Timestamp(r.date).strftime('%Y-%m-%d')
            fig.add_vline(x=date,line_color='#606060',line_dash='dot',line_width=1.5)
            fig.add_trace(go.Scatter(x=[date],y=[float(r.price)],mode='markers',
                marker=dict(symbol='circle-open',size=23,color='#111111',line=dict(width=2)),
                name='선택 거래',showlegend=False,hovertemplate='선택한 체결: %{y:,.0f}원<extra></extra>'))
            st.info(f"선택 거래: {date} · {r.action} · 체결가 {float(r.price):,.0f}원 | 신호일 {pd.Timestamp(r.signal_date):%Y-%m-%d}")
fig.update_layout(height=490,margin=dict(l=0,r=2,t=12,b=20),xaxis=dict(type='category',nticks=5,rangeslider_visible=False),yaxis=dict(side='right',tickformat=',.0f',automargin=True),legend=dict(orientation='h',y=-.16,font=dict(size=10)),hovermode='x unified',dragmode='pan',uirevision=f'{ticker}-{view}-{show}')
st.subheader('① 일봉과 매매 신호')
st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False,'scrollZoom':True,'responsive':True},key='price470')
st.caption('청산 표시는 숨겼지만 손절·위험 청산은 백테스트 수익률에 계속 반영됩니다. 기본은 깨끗한 가격 차트입니다. 위 ‘강조할 모의 체결’에서 거래를 선택하면 해당 체결 봉 하나만 강조합니다. ‘모든 체결 마커 표시’는 필요할 때만 켜세요. 신호일 종가 → 다음 거래일 시가 체결.')
st.subheader('② 스토캐스틱 (보조 참고만)')
tabs=st.tabs(['5·3·3','10·6·6','20·12·12'])
for tab,num in zip(tabs,[5,10,20]):
    with tab:
        f=go.Figure()
        f.add_trace(go.Scatter(x=x,y=p[f'K{num}'],name='%K',line=dict(color='#d74a4a',width=2)))
        f.add_trace(go.Scatter(x=x,y=p[f'D{num}'],name='%D',line=dict(color='#169b76',width=2)))
        for y in [20,80]:f.add_hline(y=y,line_dash='dot',line_color='#999')
        f.update_layout(height=245,margin=dict(l=0,r=0,t=15,b=10),xaxis=dict(type='category',nticks=5),yaxis=dict(range=[0,100],side='right'),legend=dict(orientation='h',y=1.18),dragmode='pan')
        st.plotly_chart(f,use_container_width=True,key=f'stoch44_{num}',config={'displaylogo':False,'scrollZoom':True})
st.subheader('③ 최근 신호 / 모의 체결')
recent=d.tail(45)
recent_signals=recent[recent.buy_confirmed|recent.sell_confirmed][['Close','regime','entry_type','buy_confirmed','sell_confirmed','buy_score']].tail(12).copy()
recent_signals.columns=['종가','국면','매수 유형','매수 확정','매도 확정','매수 점수']
st.dataframe(recent_signals.iloc[::-1],use_container_width=True)
if visible_ev.empty:st.write('모의 체결 기록이 없습니다. 후보와 실제 체결은 다릅니다.')
else:
    t=visible_ev.tail(20).iloc[::-1].copy()
    for col in ['date','signal_date']:t[col]=pd.to_datetime(t[col]).dt.strftime('%Y-%m-%d')
    t['price']=t.price.round(2)
    st.dataframe(t,use_container_width=True,hide_index=True)
st.subheader('④ 백테스트 요약')
if len(eq):
    dd=(eq/eq.cummax()-1).min()*100
    c1,c2,c3=st.columns(3)
    c1.metric('총수익률',f'{(eq.iloc[-1]-1)*100:.2f}%')
    c2.metric('최대낙폭',f'{dd:.2f}%')
    c3.metric('현재 보유','예' if opened else '아니오')
st.warning('교육·연구용입니다. 매수는 지지 후 확인을 기다리므로 바닥보다 늦을 수 있으며, 매도는 저항 반응을 기다리므로 고점을 놓칠 수 있습니다. 청산 기록을 숨겨도 백테스트 계산에서는 제외되지 않습니다.  실제 주문 없음. 수수료는 편도 적용하며 슬리피지·세금·갭·미체결·배당 등은 완전히 반영되지 않습니다. 특정 고점/저점을 사후에 맞추는 방식이 아니며, 과거 성과는 미래 수익을 보장하지 않습니다.')
