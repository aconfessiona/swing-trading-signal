"""Swing 4.4 mobile research dashboard; no brokerage integration."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from engine import prepare,simulate

st.set_page_config(page_title='스윙 4.4',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding:0.7rem 0.6rem 2rem;max-width:1200px}h1{font-size:1.65rem!important}h2,h3{font-size:1.2rem!important}div[data-testid="stPlotlyChart"]{overflow:hidden}@media(max-width:600px){.stRadio>div{gap:.3rem}}</style>''',unsafe_allow_html=True)
st.title('📈 스윙 매매 4.4.1')
st.caption('가격·지지저항 중심 | 거래량 가점 | 30/30/40 분할 매도 | 연구용')
with st.expander('⚙️ 종목 및 백테스트 설정',expanded=True):
    ticker=st.text_input('종목코드',value='373220.KS').strip().upper()
    a,b=st.columns(2)
    years=a.selectbox('데이터 기간',[2,3,5],index=1)
    view=b.selectbox('차트 표시',['3개월','6개월','1년','전체'],index=1)
    c,e=st.columns(2)
    fee=c.number_input('편도 비용 (bp)',min_value=0,max_value=200,value=15,step=5)
    hold=e.slider('최대 보유 거래일',10,80,40)
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
last=d.iloc[-1]
st.info(f'최근 일봉: {d.index[-1]:%Y-%m-%d} | 종가 {last.Close:,.0f} | 매수 후보 {"예" if last.buy_candidate else "아니오"} | 매수 확정 {"예" if last.buy_confirmed else "아니오"}')
st.caption('당일 종가 확정 후 신호 평가. 실시간 장중 데이터는 확정 신호가 아닙니다.')
show=st.radio('차트 표시',['핵심만','지지·저항 추가','모든 지표'],horizontal=True)
markers=st.checkbox('후보 신호 표시',value=False)
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
def mark(dates,values,name,color,symbol,size=11,outline=False):
    if len(dates)==0:return
    fig.add_trace(go.Scatter(x=pd.DatetimeIndex(pd.to_datetime(dates)).strftime('%Y-%m-%d').tolist(),y=values,mode='markers',name=name,marker=dict(symbol=symbol,size=size,color='rgba(255,255,255,0.15)' if outline else color,line=dict(color=color,width=2 if outline else 1)),hovertemplate='%{x}<br>'+name+'<extra></extra>'))
if markers:
    buy=p[p.buy_candidate];sell=p[p.sell_candidate]
    mark(buy.index,buy.Low*.98,'매수 관심','#13a074','circle-open',8,True)
    mark(sell.index,sell.High*1.02,'매도 관심','#d99a2b','circle-open',8,True)
confirmed=p[p.buy_confirmed];mark(confirmed.index,confirmed.Low*.965,'매수 확정','#13995c','triangle-up-open',12,True)
if not ev.empty:
    for action,color,symbol in [('매수','#13995c','triangle-up'),('1차 30% 매도','#dfaa25','circle'),('2차 30% 매도','#ed7e2b','diamond'),('잔여 청산','#d44949','x')]:
        e=ev[(ev.action==action)&(ev.date.isin(p.index))]
        if e.empty:continue
        ys=[float(p.loc[t,'Low']*.945 if action=='매수' else p.loc[t,'High']*1.055) for t in e.date]
        mark(e.date,ys,'모의 '+action,color,symbol,14)
fig.update_layout(height=510,margin=dict(l=0,r=2,t=10,b=15),xaxis=dict(type='category',nticks=5,rangeslider_visible=False),yaxis=dict(side='right',tickformat=',.0f',automargin=True),legend=dict(orientation='h',y=-.16,font=dict(size=10)),hovermode='x unified',dragmode='pan',uirevision=f'{ticker}-{view}-{show}')
st.subheader('① 일봉과 매매 신호')
st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False,'scrollZoom':True,'responsive':True},key='price44')
st.caption('빈 마커=조건 후보·확정, 채운 마커=실제 모의 체결. 체결은 신호 다음 거래일 시가. 후보 표시는 기본 꺼짐.')
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
recent_signals=recent[recent.buy_confirmed|recent.sell_confirmed][['Close','buy_confirmed','sell_confirmed','buy_score']].tail(12).copy()
recent_signals.columns=['종가','매수 확정','매도 확정','매수 점수']
st.dataframe(recent_signals.iloc[::-1],use_container_width=True)
if ev.empty:st.write('모의 체결 기록이 없습니다. 후보와 실제 체결은 다릅니다.')
else:
    t=ev.tail(20).iloc[::-1].copy()
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
st.warning('교육·연구용입니다. 실제 주문 없음. 수수료는 편도 적용하며 슬리피지·세금·갭·미체결·배당 등은 완전히 반영되지 않습니다. 특정 고점/저점을 사후에 맞추는 방식이 아니며, 과거 성과는 미래 수익을 보장하지 않습니다.')
