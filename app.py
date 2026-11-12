"""Mobile-first Streamlit daily swing dashboard (research only)."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from engine import indicators, signals, backtest, summary

st.set_page_config(page_title='스윙 매매 신호 3.1', page_icon='📈', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.block-container {padding-top:1rem; padding-bottom:2rem; max-width:1350px}
[data-testid="stMetricValue"] {font-size:clamp(1.1rem,3vw,1.7rem)}
@media(max-width:700px){.block-container{padding-left:.65rem;padding-right:.65rem}
 h1{font-size:1.8rem!important} h2{font-size:1.35rem!important}
 [data-testid="stHorizontalBlock"]{gap:.35rem}}
</style>''', unsafe_allow_html=True)
st.title('📈 스윙 매매 신호 3.1')
st.caption('모바일 차트 개선 · A 상승추세 눌림목 / B 과매도 반등 · 기술적 청산 · 연구용 / 자동주문 없음')

with st.expander('⚙️ 종목 및 백테스트 설정', expanded=True):
    ticker=st.text_input('종목 코드 (삼성SDI 예: 006400.KS)',value='006400.KS').strip().upper()
    c1,c2=st.columns(2)
    with c1: years=st.selectbox('데이터 조회 기간 (년)',[2,3,5],index=1)
    with c2: view=st.selectbox('차트 표시 기간',['1개월','3개월','6개월','1년','전체'],index=1)
    with st.expander('거래비용 및 보유기간'):
        fee=st.number_input('매수·매도 각 거래비용 (bp)',min_value=0,max_value=200,value=15,step=5)
        hold=st.slider('최대 보유 거래일',5,40,20)
        st.caption('15bp=편도 0.15%. 실제 수수료·세금·슬리피지와 다를 수 있습니다.')

@st.cache_data(ttl=1800,show_spinner=False)
def load(symbol,years):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    start=end-dt.timedelta(days=366*years)
    raw=yf.download(symbol,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=False)
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    if raw.empty: raise ValueError('종목 데이터가 없습니다. 코드를 확인해 주세요.')
    raw.index=pd.to_datetime(raw.index)
    if raw.index.tz is not None: raw.index=raw.index.tz_localize(None)
    return raw

try:
    if not ticker: st.warning('종목 코드를 입력하세요.'); st.stop()
    data=signals(indicators(load(ticker,years)))
    required={'RSI14','ATR14','MA20_gap_pct','A','B','exit','warning'}
    missing=required.difference(data.columns)
    if missing: raise ValueError(f'지표 열 누락: {sorted(missing)}. engine.py를 같은 버전으로 교체해 주세요.')
    valid=data.dropna(subset=['MA60','K20','D20']).copy()
    if len(valid)<30: raise ValueError('지표 계산에 필요한 일봉이 부족합니다.')
except Exception as exc:
    st.error(f'데이터 조회/계산 오류: {exc}')
    st.info('Yahoo Finance가 일시적으로 응답하지 않을 수 있습니다. 코드 또는 네트워크를 확인해 주세요.')
    st.stop()

last=valid.iloc[-1]
st.info(f'최신 수신 일봉 {valid.index[-1]:%Y-%m-%d} · 종가 {last["Close"]:,.0f} · 종가 확정 여부 별도 확인 필요')
if bool(last['A']): label='A 매수 후보'
elif bool(last['B']): label='B 매수 후보'
elif bool(last['exit']): label='기술적 청산 조건 (보유 중인 경우)'
elif bool(last['warning']): label='주의 / 관망'
else: label='관망'
st.subheader(f'최신 기술적 판단: {label}')
st.caption('종가 기준 신호입니다. 미확정 장중 일봉은 최종 신호가 아닙니다. 매매 신호는 추천이 아닙니다.')
m1,m2=st.columns(2)
m1.metric('20일 이동평균',f'{last["MA20"]:,.0f}')
m2.metric('60일 이동평균',f'{last["MA60"]:,.0f}')
st.caption(f'RSI(14): {last["RSI14"]:.1f} · 20일선 이격도: {last["MA20_gap_pct"]:+.1f}% · ATR(14): {last["ATR14"]:,.0f}')

n={'1개월':23,'3개월':65,'6개월':130,'1년':260,'전체':len(valid)}[view]
plot=valid.tail(n)
# Category dates avoid weekend gaps, make candle widths readable on iPhone.
x=plot.index.strftime('%Y-%m-%d').tolist()
plot=plot.copy(); plot['X']=x

st.subheader('① 일봉 차트')
c1,c2=st.columns(2)
with c1: overlay=st.multiselect('표시할 선',['볼린저밴드','5일선','10일선','20일선','60일선'],default=['볼린저밴드','20일선','60일선'])
with c2: marker_mode=st.selectbox('매매 신호 표시',['최근 신호만','전체 신호','숨기기'],index=0)
show_vol=st.checkbox('거래량 차트 표시',value=False)

palette={'5일선':'#17a673','10일선':'#f29d38','20일선':'#9b59b6','60일선':'#3467a3'}
fig=go.Figure()
fig.add_trace(go.Candlestick(x=x,open=plot['Open'],high=plot['High'],low=plot['Low'],close=plot['Close'],name='일봉',increasing_line_color='#d94848',decreasing_line_color='#2879cb',increasing_fillcolor='#d94848',decreasing_fillcolor='#2879cb'))
if '볼린저밴드' in overlay:
    fig.add_trace(go.Scatter(x=x,y=plot['BB_upper'],mode='lines',name='BB 상단',line=dict(color='#7889b4',width=1.5)))
    fig.add_trace(go.Scatter(x=x,y=plot['BB_lower'],mode='lines',name='BB 하단',line=dict(color='#7889b4',width=1.5),fill='tonexty',fillcolor='rgba(120,137,180,.08)'))
for name,col in [('5일선','MA5'),('10일선','MA10'),('20일선','MA20'),('60일선','MA60')]:
    if name in overlay: fig.add_trace(go.Scatter(x=x,y=plot[col],mode='lines',name=name,line=dict(color=palette[name],width=2 if name in ['20일선','60일선'] else 1.5)))
if marker_mode!='숨기기':
    for col,title,color,symbol in [('A','A 매수','#13a35e','triangle-up'),('B','B 매수','#2675d8','triangle-up'),('exit','청산 조건','#d84343','x')]:
        selected=plot[plot[col].fillna(False)]
        if marker_mode=='최근 신호만': selected=selected.tail(3)
        fig.add_trace(go.Scatter(x=selected['X'],y=selected['Low']*.985 if col!='exit' else selected['High']*1.015,
          mode='markers',name=title,marker=dict(symbol=symbol,size=12,color=color,line=dict(width=1,color='white')),
          customdata=selected.index.strftime('%Y-%m-%d'),hovertemplate='%{customdata}<br>'+title+'<extra></extra>'))
fig.update_layout(height=520,margin=dict(l=4,r=4,t=12,b=20),xaxis_rangeslider_visible=False,
    xaxis=dict(type='category',tickangle=0,nticks=6,showgrid=False),
    yaxis=dict(side='right',tickformat=',.0f',fixedrange=False),
    legend=dict(orientation='h',y=-.14,x=0,font=dict(size=11)),
    hovermode='x unified',dragmode='pan',uirevision=f'{ticker}-{view}-price',
    font=dict(size=12),showlegend=True)
st.plotly_chart(fig,key='price_main',use_container_width=True,config={'displayModeBar':True,'scrollZoom':True,'responsive':True,'displaylogo':False})
st.caption('X는 포지션과 무관한 기술적 청산 조건입니다. 백테스트 실제 청산일과 다를 수 있습니다.')
st.caption('차트에서 두 손가락으로 확대하거나 손가락으로 좌우 이동할 수 있습니다. 오른쪽 위 도구의 집 아이콘으로 원래 범위로 돌아갑니다. 매도 X는 실제 체결이 아니라 조건 발생일입니다.')

if show_vol and 'Volume' in plot:
    vfig=go.Figure(go.Bar(x=x,y=plot['Volume'],marker_color=['#d94848' if c>=o else '#2879cb' for o,c in zip(plot['Open'],plot['Close'])],name='거래량'))
    vfig.update_layout(height=180,margin=dict(l=4,r=4,t=5,b=20),xaxis=dict(type='category',nticks=6),yaxis=dict(side='right'),showlegend=False,dragmode='pan',uirevision=f'{ticker}-{view}-volume')
    st.plotly_chart(vfig,key='volume_main',use_container_width=True,config={'displayModeBar':False,'scrollZoom':True})

st.subheader('② 슬로우 스토캐스틱')
st.caption('각각 독립된 차트로 확대 표시합니다. 빨강 %K / 초록 %D, 점선 20·80 기준')
tab1,tab2,tab3=st.tabs(['단기 5·3·3','중기 10·6·6','장기 20·12·12'])
def stochastic_chart(period, location):
    chart=go.Figure()
    chart.add_trace(go.Scatter(x=x,y=plot[f'K{period}'],name='%K',mode='lines',line=dict(color='#df4646',width=2.4)))
    chart.add_trace(go.Scatter(x=x,y=plot[f'D{period}'],name='%D',mode='lines',line=dict(color='#16945c',width=2.4)))
    for y in (20,80): chart.add_hline(y=y,line_dash='dot',line_color='#8a8a8a',line_width=1)
    chart.update_layout(height=330,margin=dict(l=5,r=5,t=12,b=10),xaxis=dict(type='category',nticks=6),
        yaxis=dict(range=[0,100],side='right',tickvals=[0,20,50,80,100]),
        legend=dict(orientation='h',y=1.13,x=0),hovermode='x unified',dragmode='pan',
        uirevision=f'{ticker}-{view}-{period}')
    st.plotly_chart(chart,key=f'stoch_{location}_{period}',use_container_width=True,config={'displayModeBar':True,'scrollZoom':True,'displaylogo':False})
    st.caption(f'최신 값 · %K {last[f"K{period}"]:.1f} / %D {last[f"D{period}"]:.1f}')
with tab1: stochastic_chart(5, "tab")
with tab2: stochastic_chart(10, "tab")
with tab3: stochastic_chart(20, "tab")

with st.expander('📊 세 가지 스토캐스틱을 한 화면에서 보기'):
    for period in (5,10,20):
        st.markdown(f'**Slow ({period},{3 if period==5 else 6 if period==10 else 12},{3 if period==5 else 6 if period==10 else 12})**')
        stochastic_chart(period, "combined")

st.subheader('③ 최근 신호 발생 기록')
cols=['A','B','warning','exit']
events=valid[valid[cols].any(axis=1)].tail(15)
rows=[]
for date,row in events.iloc[::-1].iterrows():
    labels=[]
    if row['A']: labels.append('A 매수 후보')
    if row['B']: labels.append('B 매수 후보')
    if row['exit']: labels.append('기술적 청산 조건')
    if row['warning'] and not labels: labels.append('주의')
    rows.append({'날짜':date.strftime('%Y-%m-%d'),'종가':round(float(row['Close']),2),'신호':' / '.join(labels)})
if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
else: st.write('최근 발생한 신호가 없습니다.')
st.caption('주의/청산 신호는 실제 보유 여부와 무관하게 계산한 기술적 조건입니다.')

st.subheader('④ A/B 전략별 백테스트')
results=[]
for strategy in ('A','B'):
    trades,equity=backtest(valid,strategy,fee,hold)
    metrics=summary(trades,equity)
    results.append({'전략':strategy,'총수익률 (%)':round(metrics['total'],2),'최대낙폭 (%)':round(metrics['mdd'],2),'완료 거래':metrics['count'],'승률 (%)':round(metrics['win'],1)})
    with st.expander(f'{strategy} 거래 내역 ({len(trades)}건)'):
        if trades.empty: st.write('완료 거래 없음')
        else: st.dataframe(trades.sort_values('exit_date',ascending=False),hide_index=True,use_container_width=True)
st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True)
st.warning('이 프로그램은 연구용입니다. 다음 거래일 시가 체결을 가정한 과거 시뮬레이션이며 미래 성과를 보장하지 않습니다. 수정주가, 데이터 지연, 갭·슬리피지·실제 세금 및 체결 위험을 확인하세요.')
