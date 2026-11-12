"""Mobile-first Swing 4.3; research only, no trading orders."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from engine import strategy,backtest
st.set_page_config(page_title='스윙 신호 4.3',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('<style>.block-container{padding-top:.8rem;max-width:1200px}@media(max-width:700px){.block-container{padding-left:.5rem;padding-right:.5rem}h1{font-size:1.55rem!important}}</style>',unsafe_allow_html=True)
st.title('📈 스윙 매매 신호 4.3')
st.caption('볼린저 하단 반등 · 높아진 저점 · 상승 눌림 · 장기 박스권 | 연구용, 자동주문 없음')
with st.expander('종목 및 전략 설정',expanded=True):
    ticker=st.text_input('종목코드',value='373220.KS').strip().upper()
    a,b=st.columns(2);years=a.selectbox('데이터 기간(년)',[3,5,7],index=2);view=b.selectbox('차트 구간',['3개월','6개월','1년','전체'],index=1)
    a,b=st.columns(2);vf=a.slider('거래량 배수',1.0,2.5,1.3,.1);rr=b.slider('최소 예상 손익비',1.0,3.0,1.5,.25)
    a,b=st.columns(2);fee=a.number_input('편도 비용(bp)',0,200,15);hold=b.slider('최대 보유 거래일',10,60,30)
    show_candidate=st.checkbox('매수 후보/확정 조건도 표시',False)
    show_lines=st.checkbox('저항선 및 60일선 표시',False)
@st.cache_data(ttl=1800,show_spinner=False)
def load(symbol,years):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    raw=yf.download(symbol,start=str(end-dt.timedelta(days=years*366)),end=str(end),auto_adjust=True,progress=False,threads=False)
    if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
    if raw.empty:raise ValueError('가격 데이터가 없습니다.')
    raw.index=pd.to_datetime(raw.index)
    if raw.index.tz is not None:raw.index=raw.index.tz_localize(None)
    return raw
uploaded=st.file_uploader('선택: 일봉 CSV (Date,Open,High,Low,Close,Volume)',type='csv')
try:
    if uploaded:
        raw=pd.read_csv(uploaded);raw.columns=[str(c).strip().capitalize() for c in raw.columns]
        raw['Date']=pd.to_datetime(raw['Date']);raw=raw.set_index('Date')
    else:raw=load(ticker,years)
    d=strategy(raw,min_rr=rr,volume_factor=vf)
    if len(d)<140:raise ValueError('최소 140거래일 이상의 일봉이 필요합니다.')
except Exception as e:st.error(f'데이터 또는 계산 오류: {e}');st.stop()
ev,equity,opened=backtest(d,fee_bps=fee,max_hold=hold)
n={'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[view]
p=d.tail(n);x=p.index.strftime('%Y-%m-%d').tolist()
st.info(f'마지막 일봉 {d.index[-1]:%Y-%m-%d} · 종가 {d.Close.iloc[-1]:,.0f} · 최근 확정 조건 {"있음" if d.confirmed.iloc[-1] else "없음"}')
st.caption('월봉 20·12·12 계산에는 수년의 월봉이 필요합니다. 부족하면 월봉 가산점은 0점이며 주봉은 완료된 봉만 반영합니다.')
st.subheader('① 일봉과 모의 체결')
fig=go.Figure()
fig.add_trace(go.Candlestick(x=x,open=p.Open,high=p.High,low=p.Low,close=p.Close,name='일봉',increasing_line_color='#d64d4d',decreasing_line_color='#2b79c2'))
for col,name,color,width in [('BB_upper','BB 상단','#8899bb',1.2),('BB_lower','BB 하단','#8899bb',1.2),('MA20','20일선','#9753b0',2.1)]+([('MA60','60일선','#2b6399',2),('prior_high','이전 고점','#bd8b43',1.3)] if show_lines else []):
    fig.add_trace(go.Scatter(x=x,y=p[col],name=name,mode='lines',line=dict(color=color,width=width)))
def markers(frame,name,color,symbol,where,offset=1):
    if frame.empty:return
    dates=frame.index.strftime('%Y-%m-%d').tolist()
    y=frame.Low*.975 if where=='low' else frame.High*1.025
    fig.add_trace(go.Scatter(x=dates,y=y,mode='markers',name=name,marker=dict(color=color,symbol=symbol,size=12,line=dict(color='white',width=1)),hovertemplate='%{x}<br>'+name+'<extra></extra>'))
if show_candidate:
    markers(p[p.candidate_display],'매수 후보','#94a3b8','circle-open','low')
    markers(p[p.confirmed_display],'확정 조건','#16a34a','triangle-up-open','low')
if not ev.empty:
    for action,name,color,symbol,where in [('매수','모의 매수','#159967','triangle-up','low'),('1차 50% 매도','50% 매도','#e2a22c','circle','high'),('잔여/전량 청산','청산','#d44747','x','high')]:
        rows=ev[(ev.action==action)&ev.date.isin(p.index)]
        if rows.empty:continue
        prices=p.loc[rows.date]
        ys=prices.Low.to_numpy()*.965 if where=='low' else prices.High.to_numpy()*1.035
        fig.add_trace(go.Scatter(x=rows.date.dt.strftime('%Y-%m-%d'),y=ys,mode='markers',name=name,marker=dict(color=color,symbol=symbol,size=14),customdata=rows.reason,hovertemplate='%{x}<br>'+name+'<br>%{customdata}<extra></extra>'))
fig.update_layout(height=520,margin=dict(l=0,r=0,t=8,b=15),xaxis=dict(type='category',nticks=5,rangeslider_visible=False),yaxis=dict(side='right',tickformat=',.0f'),legend=dict(orientation='h',y=-.14,font=dict(size=10)),dragmode='pan',hovermode='x unified')
st.plotly_chart(fig,use_container_width=True,key='v43price',config={'displaylogo':False,'scrollZoom':True})
st.caption('마커는 모의 체결일에 표시. 조건은 전일 종가에 판단하고 다음 거래일 시가에 체결. 후보/확정 조건은 실제 체결이 아닙니다.')
st.subheader('② 세 가지 스토캐스틱')
tabs=st.tabs(['5·3·3','10·6·6','20·12·12'])
for tab,n in zip(tabs,[5,10,20]):
    with tab:
        f=go.Figure()
        for col,name,color in [(f'K{n}','%K','#d64d4d'),(f'D{n}','%D','#169b76')]:f.add_trace(go.Scatter(x=x,y=p[col],name=name,line=dict(color=color,width=2)))
        for v in [20,80]:f.add_hline(y=v,line_dash='dot',line_color='#999')
        f.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=10),xaxis=dict(type='category',nticks=5),yaxis=dict(range=[0,100],side='right'),legend=dict(orientation='h',y=1.2))
        st.plotly_chart(f,use_container_width=True,key=f'stoch{n}')
st.subheader('③ 최근 후보 및 매매 내역')
candidates=d[d.candidate_display][['pattern','score','vol_ratio','RR','confirmed']].tail(25).iloc[::-1].copy()
candidates.index=candidates.index.strftime('%Y-%m-%d')
st.caption('최근 매수 후보 25개 (모의 매수와 다름)')
st.dataframe(candidates,use_container_width=True)
st.caption('실제 모의 체결 (최근순)')
if ev.empty:st.write('현재 설정에서 모의 체결 없음')
else:
    table=ev.iloc[::-1].copy()
    for col in ['signal_date','date']:table[col]=table[col].dt.strftime('%Y-%m-%d')
    st.dataframe(table,use_container_width=True,hide_index=True)
st.subheader('④ 연구용 백테스트')
if len(equity):
    mdd=(equity/equity.cummax()-1).min()*100
    a,b,c=st.columns(3);a.metric('총수익률',f'{(equity.iloc[-1]-1)*100:.1f}%');b.metric('최대낙폭',f'{mdd:.1f}%');c.metric('모의 매수 횟수',str((ev.action=='매수').sum()) if not ev.empty else '0')
st.warning('매집은 가격·거래량 패턴의 추정일 뿐 실제 매수 주체를 확인한 것이 아닙니다. 이 전략은 실거래 검증 전 연구용이며, 수익을 보장하지 않습니다. 슬리피지·갭·데이터 편향에 유의하세요.')
