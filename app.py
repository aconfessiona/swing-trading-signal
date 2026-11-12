"""Swing 4.4 mobile research dashboard; no brokerage integration."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from engine import prepare,simulate

st.set_page_config(page_title='스윙 4.7',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding:0.7rem 0.6rem 2rem;max-width:1200px}h1{font-size:1.65rem!important}h2,h3{font-size:1.2rem!important}div[data-testid="stPlotlyChart"]{overflow:hidden}@media(max-width:600px){.stRadio>div{gap:.3rem}}</style>''',unsafe_allow_html=True)
st.title('📈 스윙 매매 4.7.0')
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
last=d.iloc[-1]
st.info(f'현재 국면 {last.regime} | 매수 제한 {"예" if last.buy_blocked else "아니오"} | 최근 일봉: {d.index[-1]:%Y-%m-%d} | 종가 {last.Close:,.0f} | 매수 후보 {"예" if last.buy_candidate else "아니오"} | 매수 확정 {"예" if last.buy_confirmed else "아니오"}')
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
