"""Mobile-friendly Swing 4.1 research dashboard. No brokerage orders."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from engine import version41,backtest_v41

st.set_page_config(page_title='스윙 매매 신호 4.1',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding:0.7rem .65rem 2rem;max-width:1200px}h1{font-size:1.7rem!important}h2,h3{font-size:1.2rem!important}[data-testid="stMetricValue"]{font-size:1.3rem}</style>''',unsafe_allow_html=True)
st.title('📈 스윙 매매 신호 4.1')
st.caption('균형형 · 매수 후보와 확정 조건 구분 · 실제 백테스트 체결 별도 · 자동주문 없음')
with st.expander('⚙️ 종목 및 전략 설정',expanded=True):
    ticker=st.text_input('종목 코드 (LG에너지솔루션: 373220.KS)','373220.KS').strip().upper()
    a,b=st.columns(2)
    years=a.selectbox('데이터 기간(년)',[2,3,5],index=1)
    view=b.selectbox('차트 기간',['3개월','6개월','1년','전체'],index=1)
    a,b=st.columns(2)
    fee=a.number_input('편도 거래비용(bp)',min_value=0,max_value=200,value=15,step=5)
    max_hold=b.slider('최대 보유 거래일',10,60,20)
    a,b=st.columns(2)
    atr_multiple=a.slider('ATR 보호 배수',1.5,4.,2.5,.25)
    min_profit=b.slider('1차 매도 최소 수익(%)',0.,10.,2.,.5)
    strategy=st.radio('실제 백테스트 포지션 전략',['A','B'],horizontal=True)
    show_candidates=st.checkbox('매수 후보 표시',True)
    show_confirmed=st.checkbox('확정 매수 조건 표시',True)
    show_executions=st.checkbox('실제 체결 표시',True)

@st.cache_data(ttl=1800,show_spinner=False)
def load(symbol,yrs):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    start=end-dt.timedelta(days=366*yrs)
    df=yf.download(symbol,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=False)
    if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
    if df.empty:raise ValueError('가격 데이터가 없습니다. 종목 코드를 확인하세요.')
    df.index=pd.to_datetime(df.index)
    if df.index.tz is not None:df.index=df.index.tz_localize(None)
    return df
uploaded=st.file_uploader('선택: 일봉 CSV (Date,Open,High,Low,Close,Volume)',type=['csv'])
try:
    if uploaded:
        raw=pd.read_csv(uploaded);raw.columns=[str(x).strip().capitalize() for x in raw.columns]
        if 'Date' not in raw:raise ValueError('CSV에는 Date 열이 필요합니다.')
        raw['Date']=pd.to_datetime(raw.Date);raw=raw.set_index('Date')
    else:raw=load(ticker,years)
    d=version41(raw).dropna(subset=['MA60','K20','D20']).copy()
    if len(d)<85:raise ValueError('최소 85개 이상의 지표 계산 후 거래일이 필요합니다.')
except Exception as exc:
    st.error(f'데이터 또는 지표 오류: {exc}');st.stop()

last=d.iloc[-1]
st.info(f'최신 일봉 {d.index[-1]:%Y-%m-%d} · 종가 {last.Close:,.0f} · A 후보 {"예" if last.A_candidate else "아니오"} / 확정 {"예" if last.A else "아니오"} · B 후보 {"예" if last.B_candidate else "아니오"} / 확정 {"예" if last.B else "아니오"}')
st.caption('후보와 확정은 당일 종가 기준 조건입니다. 확정 조건은 반드시 실제 매수를 뜻하지 않습니다. 체결은 보유 상태를 고려한 다음 거래일 시가 모의 결과입니다.')
aev,aeq,aopen=backtest_v41(d,'A',fee,max_hold,atr_multiple,min_profit)
bev,beq,bopen=backtest_v41(d,'B',fee,max_hold,atr_multiple,min_profit)
ev= aev if strategy=='A' else bev
n={'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[view]
p=d.tail(n).copy();x=p.index.strftime('%Y-%m-%d').tolist()
st.subheader('① 가격 · 매수 후보 · 확정 · 모의 체결')
with st.expander('차트 선 선택',expanded=False):
    choices=['볼린저밴드','20일선','60일선','상승 채널','이전 60일 고점','주봉 저항','거래량 집중 근사']
    lines=st.multiselect('표시 선',choices,default=['볼린저밴드','20일선','60일선'])
fig=go.Figure()
fig.add_trace(go.Candlestick(x=x,open=p.Open,high=p.High,low=p.Low,close=p.Close,name='일봉',increasing_line_color='#d84a4a',decreasing_line_color='#2879cb',increasing_fillcolor='#d84a4a',decreasing_fillcolor='#2879cb'))
def addline(col,name,color,dash='solid',width=1.6):
    fig.add_trace(go.Scatter(x=x,y=p[col],mode='lines',name=name,line=dict(color=color,dash=dash,width=width)))
if '볼린저밴드' in lines:
    addline('BB_upper','BB 상단','#8595b8');addline('BB_lower','BB 하단','#8595b8')
if '20일선' in lines:addline('MA20','20일선','#9754af',width=2.2)
if '60일선' in lines:addline('MA60','60일선','#356ca2',width=2.2)
if '상승 채널' in lines:
    addline('channel_upper','채널 상단','#ce9637','dot');addline('channel_lower','채널 하단','#198f6d','dot')
if '이전 60일 고점' in lines:addline('prior_high_60','60일 고점','#d17a43','dash')
if '주봉 저항' in lines:
    addline('week_prior_high','주봉 고점','#ae715a','dash');addline('week_upper','주봉 BB','#bb8a42','dot')
if '거래량 집중 근사' in lines:addline('volume_resistance','거래량 근사','#666','dash')

def markers(frame,name,color,symbol,above=False,size=10):
    if frame.empty:return
    ds=frame.index.strftime('%Y-%m-%d').tolist()
    ys=(frame.High*1.025 if above else frame.Low*.975).tolist()
    fig.add_trace(go.Scatter(x=ds,y=ys,mode='markers',name=name,marker=dict(color=color,symbol=symbol,size=size,line=dict(color='white',width=1)),hovertemplate='%{x}<br>'+name+'<extra></extra>'))
if show_candidates:markers(p[p[f'{strategy}_candidate']&~p[strategy]],f'{strategy} 매수 후보','#9cabc1','circle-open',9)
if show_confirmed:markers(p[p[strategy]],f'{strategy} 확정 조건','#18a777','triangle-up-open',12)
if show_executions and not ev.empty:
    for action,name,color,symbol in [('매수','실제 모의 매수','#0b8751','triangle-up'),('1차 매도','모의 50% 매도','#e3a21a','circle'),('전량 청산','모의 전량 청산','#d54242','x')]:
        selected=ev[(ev.action==action)&(ev.date.isin(p.index))]
        if selected.empty:continue
        # Actual execution date and execution price, not signal day price.
        fig.add_trace(go.Scatter(x=selected.date.dt.strftime('%Y-%m-%d'),y=selected.price,mode='markers',name=name,marker=dict(color=color,symbol=symbol,size=14,line=dict(color='white',width=1)),customdata=selected[['reason','signal_date']].astype(str).to_numpy(),hovertemplate='%{x}<br>'+name+'<br>체결가 %{y:,.0f}<br>%{customdata[0]}<br>신호일 %{customdata[1]}<extra></extra>'))
fig.update_layout(height=530,margin=dict(l=0,r=0,t=12,b=20),xaxis=dict(type='category',nticks=5,rangeslider_visible=False),yaxis=dict(side='right',tickformat=',.0f'),legend=dict(orientation='h',y=-.15,font=dict(size=10)),dragmode='pan',hovermode='closest',uirevision=f'{ticker}-{view}-{strategy}')
st.plotly_chart(fig,key='v41_price',use_container_width=True,config={'displaylogo':False,'scrollZoom':True,'responsive':True})
st.caption('○ 후보: 조건 일부 충족 / △ 테두리: 종가 기준 확정 조건 / 채운 마커: 포지션 상태를 고려한 다음 시가 모의 체결. 서로 다른 개념입니다.')
st.subheader('② 세 가지 스토캐스틱')
for tab,period in zip(st.tabs(['5·3·3','10·6·6','20·12·12']),[5,10,20]):
    with tab:
        f=go.Figure()
        f.add_trace(go.Scatter(x=x,y=p[f'K{period}'],name='%K',line=dict(color='#d34b4b',width=2)))
        f.add_trace(go.Scatter(x=x,y=p[f'D{period}'],name='%D',line=dict(color='#169b76',width=2)))
        for v in (20,80):f.add_hline(y=v,line_dash='dot',line_color='#999')
        f.update_layout(height=290,margin=dict(l=0,r=0,t=10,b=10),xaxis=dict(type='category',nticks=5),yaxis=dict(range=[0,100],side='right'),legend=dict(orientation='h',y=1.17))
        st.plotly_chart(f,key=f'v41_stoch_{period}',use_container_width=True,config={'displaylogo':False,'scrollZoom':True})
        st.caption(f'최근 %K {last[f"K{period}"]:.1f} · %D {last[f"D{period}"]:.1f}')
st.subheader('③ 최근 매수 후보·확정 조건')
recent=d.tail(130); rec=[]
for day,r in recent.iterrows():
    for name in ('A','B'):
        if bool(r[name]):rec.append({'날짜':day.strftime('%Y-%m-%d'),'전략':name,'구분':'확정 조건','종가':round(r.Close,2)})
        elif bool(r[f'{name}_candidate']):rec.append({'날짜':day.strftime('%Y-%m-%d'),'전략':name,'구분':'후보','종가':round(r.Close,2)})
if rec:st.dataframe(pd.DataFrame(rec).iloc[::-1].head(60),hide_index=True,use_container_width=True)
else:st.write('최근 130거래일에 후보/확정 조건이 없습니다.')
st.subheader('④ 실제 모의 체결 기록')
if ev.empty:st.write('선택 전략의 모의 체결 기록이 없습니다.')
else:
    table=ev.copy()
    for col in ('signal_date','date'):table[col]=table[col].dt.strftime('%Y-%m-%d')
    table=table.rename(columns={'signal_date':'신호일','date':'체결일','action':'구분','reason':'사유','price':'체결가','fraction':'최초 수량 대비 비율'})
    st.dataframe(table.iloc[::-1],hide_index=True,use_container_width=True)
st.subheader('⑤ 전략 A/B 백테스트')
rows=[]
for name,equity,events,opened in [('A',aeq,aev,aopen),('B',beq,bev,bopen)]:
    if equity.empty:continue
    dd=(equity/equity.cummax()-1).min()*100
    count=lambda label:int((events.action==label).sum()) if not events.empty else 0
    rows.append({'전략':name,'총수익률(%)':round((equity.iloc[-1]-1)*100,2),'최대낙폭(%)':round(dd,2),'매수 횟수':count('매수'),'1차 매도':count('1차 매도'),'전량 청산':count('전량 청산'),'보유 중':opened})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.warning('연구용 백테스트입니다. 1차 매도 최소 수익 조건은 신호일 종가 기준이며 다음날 시가 갭으로 실제 체결 수익이 달라질 수 있습니다. 슬리피지·세금·미체결·유동성은 별도 검증이 필요합니다. 과거 성과는 미래 수익을 보장하지 않습니다.')
