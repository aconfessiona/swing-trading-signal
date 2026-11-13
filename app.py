import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from engine import VERSION, analyze_53, simulate_held_position_53

st.set_page_config(page_title='Swing Analysis 5.3.1',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('<style>.block-container{max-width:1100px;padding-top:1rem}@media(max-width:680px){.block-container{padding-left:.6rem;padding-right:.6rem}}[data-testid="stMetric"]{border:1px solid #e5e7eb;border-radius:12px;padding:10px}</style>',unsafe_allow_html=True)
st.title('📈 스윙 매매 분석 5.3.1 · 차트 간소화')
st.caption('볼린저 상단 · 밴드 워크 보호 · 다중 기간 저항 · 가상 30/30/40 분할 매도')
with st.expander('종목 및 차트 설정',expanded=True):
    ticker=st.text_input('종목 코드',value='373220.KS').strip().upper()
    x,y=st.columns(2)
    years=x.selectbox('수집 기간(년)',[3,5,7],index=1)
    period=y.selectbox('표시 기간',['3개월','6개월','1년','전체'],index=1)
    show_ma=st.checkbox('20·60일선',True)
    show_levels=st.checkbox('단기·중기·장기 저항',True)
    show_markers=st.checkbox('선제 익절 후보 표시 (체결 아님)',True)
    show_watch=st.checkbox('기존 지지 관심 마커(매수 아님)',False)
    show_sim=st.checkbox('가상 보유 30/30/40 매도 체결 표시',False)
@st.cache_data(ttl=1800,show_spinner=False)
def fetch(ticker,years):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    df=yf.download(ticker,start=str(end-dt.timedelta(days=years*366)),end=str(end),auto_adjust=True,progress=False,threads=False)
    if df.empty:raise ValueError('시세를 불러올 수 없습니다.')
    if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
    return df
uploaded=st.file_uploader('선택: 일봉 CSV(Date,Open,High,Low,Close,Volume)',type='csv')
try:
    if uploaded:
        raw=pd.read_csv(uploaded)
        raw.columns=[str(c).strip().capitalize() for c in raw.columns]
        raw=raw.set_index(pd.to_datetime(raw['Date'])).drop(columns=['Date'])
    else:raw=fetch(ticker,years)
    d=analyze_53(raw,ma_below_days=3,breakout_pct=.01,retest_days=10,min_rr=1.8,adx_ref=18)
except Exception as exc:
    st.error(f'데이터/분석 오류: {exc}');st.stop()
last=d.iloc[-1]
regimes={'UP':'상승','DOWN':'하락','RANGE':'횡보','WEAK':'상승 약화','REVERSAL_PREP':'전환 준비','REVERSAL_CONFIRMED':'전환 확인','NEUTRAL':'판별 유보','WARMUP':'준비'}
a,b,c=st.columns(3)
a.metric('국면',regimes.get(last.regime,last.regime))
b.metric('매수 상태','금지' if last.buy_blocked else '미검증')
c.metric('매도 판단','밴드 워크 보호' if last.strong_protect else ('반락 확인' if last.sell_confirm else '관찰' if last.sell_watch else '대기'))
st.caption(f'버전 {VERSION} · 마지막 일봉 {d.index[-1]:%Y-%m-%d} · 매수 확정 신호는 미구현')
entry_date=None;events=[]
if show_sim:
    st.info('가상 보유: 선택 날짜 다음 시가에 진입했다고 가정합니다. 실제 매수 신호가 아닙니다.')
    entry_date=st.date_input('가상 진입 판단일',value=d.index[max(60,len(d)-110)].date(),min_value=d.index[60].date(),max_value=d.index[-2].date())
    try:
        events,sim=simulate_held_position_53(d,entry_date)
        st.caption(f'가상 진입 {sim["entry"]:,.0f} · 잔여 {sim["remaining"]:.0%} · 실현손익(초기 1주 기준) {sim["realized_pnl_per_initial_share"]:,.0f} · 미실현손익 {sim["unrealized_pnl_per_initial_share"]:,.0f}')
    except Exception as exc:st.warning(str(exc))
n={'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[period]
p=d.tail(n)
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24],vertical_spacing=.035)
fig.add_trace(go.Candlestick(x=p.index,open=p.Open,high=p.High,low=p.Low,close=p.Close,name='일봉',increasing_line_color='#d05b63',decreasing_line_color='#3983b3'),row=1,col=1)
if show_ma:
    for col,name,color in [('MA20','MA20','#9473ba'),('MA60','MA60','#397ba9')]:
        fig.add_trace(go.Scatter(x=p.index,y=p[col],mode='lines',name=name,line=dict(color=color,width=1.7)),row=1,col=1)
for col,name in [('BB_upper','볼린저 상단'),('BB_lower','볼린저 하단')]:
    fig.add_trace(go.Scatter(x=p.index,y=p[col],mode='lines',name=name,line=dict(color='#aa8bc2',width=1.2,dash='dot')),row=1,col=1)
if show_levels:
    for col,name,color in [('short_resistance','단기 저항','#d0a34d'),('mid_resistance','중기 저항','#d28b91'),('long_resistance','장기 저항','#a85d70')]:
        level=last[col]
        if pd.notna(level):
            fig.add_trace(go.Scatter(x=[p.index[0],p.index[-1]],y=[level,level],mode='lines',name=name,line=dict(color=color,width=1.5,dash='dash'),hovertemplate=name+': %{y:,.0f}<extra></extra>'),row=1,col=1)
if show_markers:
    for col,name,color,symbol in [('early_exit','선제 익절 후보','#d6a34c','triangle-down')]:
        q=p[p[col]]
        if len(q):fig.add_trace(go.Scatter(x=q.index,y=q.High+q.ATR*.35,mode='markers',name=name,marker=dict(symbol=symbol,size=9,color=color),customdata=q.sell_reason,hovertemplate='%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>'),row=1,col=1)
if show_watch:
    q=p[p.zone_watch]
    if len(q):fig.add_trace(go.Scatter(x=q.index,y=q.Low-q.ATR*.4,mode='markers',name='지지 관심(매수 아님)',marker=dict(symbol='circle-open',size=8,color='#68a67b')),row=1,col=1)
if events:
    for action,color,symbol in [('가상 진입','#299d75','triangle-up'),('1차 매도 30%','#d6a34c','triangle-down'),('2차 매도 30%','#d26b76','triangle-down'),('보호 청산','#777777','x')]:
        es=[e for e in events if e['action']==action and e['date'] in p.index]
        if es:fig.add_trace(go.Scatter(x=[e['date'] for e in es],y=[e['price'] for e in es],mode='markers',name=action,marker=dict(symbol=symbol,size=13,color=color,line=dict(width=1,color='white'))),row=1,col=1)
wave=p.momentum_wave.fillna(0)
fig.add_trace(go.Bar(x=p.index,y=wave,name='모멘텀',marker_color=np.where(wave>=0,'#319a73','#ce666c')),row=2,col=1)
fig.add_hline(y=0,line_color='#a1a1aa',line_width=1,row=2,col=1)
fig.update_layout(height=620,margin=dict(l=4,r=6,t=8,b=8),xaxis_rangeslider_visible=False,yaxis=dict(side='right',tickformat=',.0f',gridcolor='rgba(130,130,130,.12)'),yaxis2=dict(side='right',showgrid=False),legend=dict(orientation='h',y=-.12,font=dict(size=10)),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',hovermode='x unified',dragmode='pan')
st.plotly_chart(fig,use_container_width=True,key='swing51chart',config={'displaylogo':False,'scrollZoom':True})
st.caption('작은 주황 삼각형=선제 익절 후보(체결 아님) · 반락 후보는 차트에서 숨김 · 큰 마커=가상 보유 체결. 저항선은 최신 기준 3개만 표시하며 과거 시점의 저항을 의미하지 않습니다.')
st.subheader('매도 판단 근거')
if last.strong_protect:st.success('밴드 워크 보호 중: 볼린저 상단 접촉만으로 매도하지 않음')
elif last.sell_confirm:st.warning('반락 확인 조건 충족: 보유 중이라면 다음 거래일 시가 기준으로 판단')
elif last.sell_watch:st.info('볼린저 상단 또는 저항 접근: 반락 대기')
else:st.info('현재 매도 관심 조건 없음')
st.write('판단 사유:',last.sell_reason or '없음')
a,b,c=st.columns(3)
for box,col,name in [(a,'short_resistance','단기 60일'),(b,'mid_resistance','중기 180일'),(c,'long_resistance','장기 756일')]:
    box.metric(name,f'{last[col]:,.0f}' if pd.notna(last[col]) else '미확인')
st.caption('현재가 위의 확정 피벗 고점 중 가장 가까운 값. 주봉·매물대는 아직 미구현입니다.')
st.subheader('선제 익절 후보 발생 내역')
selected=d[d.early_exit][['Close','BB_upper','RSI14','regime','sell_candidate_type','sell_candidate_score','sell_reason']].tail(30).iloc[::-1].copy()
st.dataframe(selected,use_container_width=True,height=230)
st.caption('후보는 가상 진입일과 무관하게 표시됩니다. 실제 체결은 포지션 보유 중일 때만 별도 마커로 표시합니다.')
st.subheader('날짜별 판단 기록')
cols=['Close','regime','buy_blocked','BB_upper','percent_b','band_walk','strong_protect','sell_watch','sell_score','early_exit','second_exit','sell_confirm','sell_reason','sell_block_reason','sell_candidate_score','short_resistance','mid_resistance','long_resistance']
record=d[cols].tail(n).iloc[::-1].copy();record.index=record.index.strftime('%Y-%m-%d')
st.dataframe(record,use_container_width=True,height=300)
st.download_button('전체 분석 CSV',d[cols].to_csv(index_label='Date').encode('utf-8-sig'),file_name='swing_5_3_1_diagnostics.csv',mime='text/csv')
if events:st.dataframe(pd.DataFrame(events),hide_index=True,use_container_width=True)
with st.expander('제한 사항'):
    st.write('매수 신호는 미구현. 가상 진입은 매수 정책 통과를 의미하지 않음. 저항은 일봉 확정 고점이며 주봉/가격별 거래량 매물대는 미구현. 전략 수익성은 아직 검증하지 않음.')
