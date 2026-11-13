"""Swing 5.0 A/B: regime and buy-block diagnostics. No trading orders."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from engine import VERSION, analyze

st.set_page_config(page_title='스윙 분석 5.0 A/B',page_icon='📊',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding-top:.7rem;max-width:1150px}@media(max-width:650px){.block-container{padding-left:.55rem;padding-right:.55rem}h1{font-size:1.5rem!important}h2{font-size:1.2rem!important}}</style>''',unsafe_allow_html=True)
st.title('📊 스윙 매매 분석 시스템 5.0 A/B')
st.caption('국면 판별 + 매수 금지 검증 전용 · 실제 주문/모의 체결 없음 · 종가 기준 분석')

with st.expander('종목 · 데이터 · 표시 설정',expanded=True):
    symbol=st.text_input('종목 코드 (한국: .KS / .KQ, 미국: AAPL)',value='373220.KS').strip().upper()
    a,b=st.columns(2)
    years=a.selectbox('데이터 기간', [3,5,7,10],index=2)
    view=b.selectbox('차트 범위',['3개월','6개월','1년','전체'],index=1)
    show_watch=st.checkbox('관심 구간 표시 (매수 아님)',value=True)
    show_ma60=st.checkbox('60일선 표시',value=True)
    st.caption('확정값: 20일선 3일 이탈 · 전고점 +1% · 재지지 10거래일 · 최소 손익비 1.8 · ADX 18')

@st.cache_data(ttl=1800,show_spinner=False)
def fetch(ticker,years):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    raw=yf.download(ticker,start=str(end-dt.timedelta(days=years*366)),end=str(end),auto_adjust=True,progress=False,threads=False)
    if raw.empty:raise ValueError('시세를 받지 못했습니다. 종목코드 또는 네트워크를 확인하세요.')
    if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
    raw.index=pd.to_datetime(raw.index)
    return raw

upload=st.file_uploader('선택: 일봉 CSV 업로드 (Date,Open,High,Low,Close,Volume)',type=['csv'])
try:
    if upload is not None:
        raw=pd.read_csv(upload)
        raw.columns=[str(c).strip().capitalize() for c in raw.columns]
        if 'Date' not in raw.columns:raise ValueError('CSV에 Date 열이 필요합니다.')
        raw['Date']=pd.to_datetime(raw['Date'])
        raw=raw.set_index('Date')
    else:
        if not symbol:st.warning('종목코드를 입력하세요.');st.stop()
        raw=fetch(symbol,years)
    d=analyze(raw,ma_below_days=3,breakout_pct=.01,retest_days=10,min_rr=1.8,adx_ref=18)
except Exception as exc:
    st.error(f'데이터 또는 분석 오류: {exc}')
    st.stop()

names={'UP':'상승','WEAK':'상승 약화','RANGE':'횡보','DOWN':'하락','REVERSAL_PREP':'전환 준비','REVERSAL_CONFIRMED':'상승 전환 확인','NEUTRAL':'판별 유보','WARMUP':'초기 계산 구간'}
colors={'UP':'rgba(38,148,105,.12)','WEAK':'rgba(222,163,44,.15)','RANGE':'rgba(72,124,195,.11)','DOWN':'rgba(208,69,69,.13)','REVERSAL_PREP':'rgba(234,163,68,.12)','REVERSAL_CONFIRMED':'rgba(29,162,128,.16)','NEUTRAL':'rgba(125,125,125,.07)','WARMUP':'rgba(125,125,125,.04)'}
last=d.iloc[-1]
st.info(f'마지막 일봉 {d.index[-1]:%Y-%m-%d} · 종가 {last.Close:,.0f} · 국면 **{names.get(last.regime,last.regime)}** · 신규 매수 기본 차단 **{"예" if last.buy_blocked else "아니오"}**')
st.caption('기본 차단 없음은 매수 신호가 아닙니다. 진입·손절·목표가와 손익비 1.8 검사는 다음 단계(5.0-C)에서 구현합니다.')

n={'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[view]
p=d.tail(n).copy();x=p.index.strftime('%Y-%m-%d').tolist()
fig=go.Figure()
# Colored background segments for regime, without placing large overlapping markers.
segments=[]
for k in range(len(p)):
    r=p.regime.iloc[k]
    if not segments or segments[-1][2]!=r:segments.append([k,k,r])
    else:segments[-1][1]=k
for a,b,r in segments:
    fig.add_vrect(x0=max(-.5,a-.5),x1=b+.5,fillcolor=colors.get(r,colors['NEUTRAL']),opacity=1,line_width=0,layer='below')
fig.add_trace(go.Candlestick(x=x,open=p.Open,high=p.High,low=p.Low,close=p.Close,name='일봉',increasing_line_color='#d34d4d',decreasing_line_color='#2879be',showlegend=True))
for col,name,color,width in [('BB_upper','BB 상단','#8b99b5',1),('BB_lower','BB 하단','#8b99b5',1),('MA20','20일선','#9052af',2.1)]+([('MA60','60일선','#37699c',2)] if show_ma60 else []):
    fig.add_trace(go.Scatter(x=x,y=p[col],mode='lines',name=name,line=dict(color=color,width=width),hoverinfo='skip'))
if show_watch:
    w=p[p.watch]
    if not w.empty:
        fig.add_trace(go.Scatter(x=w.index.strftime('%Y-%m-%d'),y=w.Low-w.ATR*.35,mode='markers',name='지지 관심 (매수 아님)',marker=dict(symbol='circle-open',size=8,color='#b18b37',line=dict(width=1.5)),customdata=w.watch_reason,hovertemplate='%{x}<br>관심: %{customdata}<extra></extra>'))
fig.update_layout(height=480,margin=dict(l=0,r=0,t=8,b=10),xaxis=dict(type='category',rangeslider_visible=False,nticks=5),yaxis=dict(side='right',tickformat=',.0f'),legend=dict(orientation='h',y=-.13,font=dict(size=10)),dragmode='pan',hovermode='closest',showlegend=True)
st.subheader('① 일봉과 국면 배경')
st.plotly_chart(fig,use_container_width=True,key='swing50ab_main',config={'displaylogo':False,'scrollZoom':True})
st.caption('배경: 녹색 상승 · 노랑 상승 약화/전환 준비 · 파랑 횡보 · 빨강 하락 · 회색 판별 유보. 빈 원은 관심 위치이며 매수가 아닙니다.')

st.subheader('② 마지막 날 매수 금지 판정')
if last.buy_blocked:st.error(last.block_reasons)
else:st.success('기본 금지 조건 없음. 매수 확정은 아님.')
if last.watch:st.write('지지 관심 근거:',last.watch_reason)

st.subheader('③ 날짜별 검증 기록')
summary=pd.DataFrame({'날짜':d.index.strftime('%Y-%m-%d'),'종가':d.Close.round(2).values,'국면':d.regime.map(names).values,'매수 금지':d.buy_blocked.map({True:'금지',False:'기본 차단 없음'}).values,'금지 이유':d.block_reasons.values,'관심':d.watch.map({True:'관심',False:''}).values,'관심 근거':d.watch_reason.values,'국면 변경 이유':d.transition_reason.values})
recent=summary.tail(n).iloc[::-1]
st.dataframe(recent,use_container_width=True,hide_index=True,height=340)
st.download_button('전체 분석 기록 CSV 다운로드',data=summary.to_csv(index=False).encode('utf-8-sig'),file_name='swing_5_0_ab_regime_diagnostics.csv',mime='text/csv',key='download_50ab')

st.subheader('④ 검증용 요약')
a,b,c=st.columns(3)
a.metric('분석 일봉',f'{len(d):,}')
b.metric('하락 국면 매수 허용 위반',str(int(((d.regime=='DOWN')&(~d.buy_blocked)).sum())))
c.metric('최근 구간 관심 일수',str(int(p.watch.sum())))
st.caption('하락 국면 차단 위반 0건은 로직 불변식 검사일 뿐, 수익률이나 시장 예측 정확도를 의미하지 않습니다.')
with st.expander('설계 제한과 다음 단계'):
    st.markdown('''- **현재 5.0-A/B는 진입 신호나 모의 거래를 만들지 않습니다.** 국면과 차단 사유를 먼저 검증합니다.
- 20일선 3일 이탈, 전고점 1% 돌파, 재지지 10일을 적용합니다. ADX 18은 참고값으로 보관하며, 방향은 가격 구조와 이동평균선으로 판단합니다.
- 저점 상승(HL) 단독 경로, 피보나치 4수준, 파동, 손익비 1.8, 30/30/40 매도는 후속 단계에서 구현합니다.
- 5봉 피벗은 오른쪽 2개 봉 완성 후 확정됩니다. 과거 날짜에 소급해서 매매하지 않습니다.
- 국면 규칙은 초기 가설이며 여러 종목·시장 국면에서 검증해야 합니다. 금융 투자 자문이나 자동 주문 기능이 아닙니다.''')
