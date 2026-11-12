import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import indicators,backtest,stats

st.set_page_config(page_title='스윙 매매 신호 1.0',layout='wide')
st.title('📈 스윙 매매 신호 1.0')
st.caption('A: 상승 추세 눌림목 | B: 과매도 반등 · 일봉 종가 신호, 다음 거래일 시가 진입 · 교육/연구용')
with st.sidebar:
    st.header('데이터')
    mode=st.radio('불러오기 방식',['종목코드 (Yahoo Finance)','CSV 업로드'])
    if mode.startswith('종목'):
        ticker=st.text_input('종목코드','006400.KS',help='삼성SDI: 006400.KS / 코스닥: .KQ / 미국: AAPL')
        years=st.slider('조회 기간 (년)',1,10,5)
    else:
        file=st.file_uploader('일봉 CSV',type='csv')
        st.caption('Date,Open,High,Low,Close,Volume 형식. 수정주가 일관성에 유의하세요.')
    st.header('백테스트')
    initial=st.number_input('초기 자산 (원)',min_value=100000,value=10000000,step=1000000)
    risk=st.slider('거래당 위험 한도 (%)',0.25,2.0,1.0,0.25)
    pos=st.slider('최대 종목 비중 (%)',5,100,30,5)
    fee=st.number_input('매수·매도 각각 비용 (bp)',min_value=0,max_value=300,value=15,help='15bp=0.15%. 수수료·세금·슬리피지를 각 체결마다 일괄 근사')
try:
    if mode.startswith('종목'):
        import yfinance as yf
        raw=yf.download(ticker,period=f'{years}y',interval='1d',auto_adjust=True,progress=False,multi_level_index=False)
        if raw.empty:
            st.error('가격 데이터를 가져오지 못했습니다. 종목코드를 확인하거나 CSV를 업로드하세요.')
            st.stop()
        raw=raw.rename_axis('Date').reset_index()
    else:
        if file is None:
            st.info('CSV를 업로드하면 분석을 시작합니다.')
            st.stop()
        raw=pd.read_csv(file)
    d=indicators(raw)
except Exception as ex:
    st.error(f'데이터 처리 오류: {ex}')
    st.stop()

trades,eq=backtest(d,initial,risk,fee,pos)
metrics=stats(trades,eq,initial)
a,b,c,e=st.columns(4)
a.metric('백테스트 수익률',f"{metrics['수익률(%)']:.2f}%")
b.metric('최대낙폭',f"{metrics['최대낙폭(%)']:.2f}%")
c.metric('완료 거래',f"{metrics['완료 거래 수']}회")
e.metric('승률',f"{metrics['승률(%)']:.1f}%")
last=d.iloc[-1]
st.subheader('최근 일봉 신호')
st.write(f"기준일: **{d.index[-1].date()}** · 종가: **{last.close:,.2f}** · 신호: **{last.signal or '관망'}**")
st.caption('신호는 종가 확정 후에만 유효합니다. 종가가 아직 확정되지 않은 장중 데이터는 최종 신호가 아닙니다.')

fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.04,row_heights=[0.75,0.25],subplot_titles=('일봉 + 볼린저밴드 + 이동평균선','스토캐스틱'))
fig.add_trace(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name='가격'),row=1,col=1)
for key,color in [('ma5','#16a34a'),('ma10','#ef4444'),('ma20','#9333ea'),('ma60','#f59e0b'),('bb10_upper','#1e3a8a'),('bb10_lower','#94a3b8')]:
    fig.add_trace(go.Scatter(x=d.index,y=d[key],mode='lines',name=key,line=dict(color=color,width=1)),row=1,col=1)
for s,color,sym in [('A','#16a34a','triangle-up'),('B','#2563eb','triangle-up')]:
    v=d[d.signal==s]
    fig.add_trace(go.Scatter(x=v.index,y=v.low*0.985,mode='markers',name=f'{s} 신호',marker=dict(symbol=sym,size=12,color=color)),row=1,col=1)
if len(trades):
    fig.add_trace(go.Scatter(x=trades.exit_date,y=trades.exit,mode='markers',name='청산 체결',marker=dict(symbol='x',size=10,color='#dc2626')),row=1,col=1)
for n,color in [(5,'#ef4444'),(10,'#2563eb'),(20,'#9333ea')]:
    fig.add_trace(go.Scatter(x=d.index,y=d[f'k{n}'],name=f'%K {n}',line=dict(color=color,width=1)),row=2,col=1)
fig.add_hline(y=80,line_dash='dot',row=2,col=1)
fig.add_hline(y=20,line_dash='dot',row=2,col=1)
fig.update_layout(height=750,xaxis_rangeslider_visible=False,legend=dict(orientation='h'),margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig,use_container_width=True)
st.subheader('자산 곡선')
st.line_chart(eq.equity)
st.subheader('완료 거래 내역')
if len(trades):
    st.dataframe(trades.sort_values('exit_date',ascending=False),use_container_width=True,hide_index=True)
    st.download_button('거래 내역 CSV 저장',trades.to_csv(index=False).encode('utf-8-sig'),'trades.csv','text/csv')
else:
    st.info('선택한 기간에 완료된 거래가 없습니다. 전략 조건이 엄격할 수 있습니다.')
st.download_button('지표·신호 CSV 저장',d.to_csv().encode('utf-8-sig'),'signals.csv','text/csv')
st.warning('주의: 이 도구는 투자 조언이나 실시간 주문 시스템이 아닙니다. 과거 성과는 미래 수익을 보장하지 않습니다. 매매비용은 단순 근사이며 유동성, 체결 지연, 상장폐지, 데이터 품질 및 기업행동의 영향을 완전히 반영하지 않습니다. 마지막 미청산 포지션은 자산곡선에 시가평가로 반영되지만 완료 거래 표에는 포함되지 않습니다.')
