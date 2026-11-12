import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from engine import indicators, signals, backtest, summary

st.set_page_config(page_title='스윙 매매 신호 2.0',page_icon='📈',layout='wide')
st.title('📈 스윙 매매 신호 2.0')
st.caption('1~4주 스윙 · A 상승추세 눌림목 / B 과매도 반등 · 기술적 매도 · 연구용 (자동주문 없음)')
with st.sidebar:
    st.header('분석 설정')
    symbol=st.text_input('종목 코드 (예: 삼성SDI 006400.KS)',value='006400.KS').strip().upper()
    years=st.selectbox('데이터 조회 기간', [2,3,5],index=1)
    view=st.selectbox('차트 표시 기간',['3개월','6개월','1년','전체'],index=1)
    fee=st.number_input('매수/매도 각 거래비용 (bp)',min_value=0,max_value=200,value=15,step=5)
    hold=st.slider('최대 보유 거래일',min_value=5,max_value=40,value=20)
    st.caption('15bp = 편도 0.15%. 세금·수수료·슬리피지를 합산한 가정값으로 조정하세요.')

@st.cache_data(ttl=1800,show_spinner=False)
def load(ticker, years):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    start=end-dt.timedelta(days=366*years)
    df=yf.download(ticker,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=False)
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    if df.empty: raise ValueError('가격 데이터를 찾지 못했습니다. 종목 코드를 확인하세요.')
    df.index=pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
    return df

try:
    raw=load(symbol,years)
    data=signals(indicators(raw))
    valid=data.dropna(subset=['MA60','K20','D20'])
    if len(valid)<30: st.error('분석 가능한 일봉 데이터가 부족합니다.'); st.stop()
except Exception as e:
    st.error(f'데이터 조회/분석 실패: {e}')
    st.info('Yahoo Finance 데이터는 지연되거나 일시적으로 차단될 수 있습니다. 잠시 후 다시 시도하세요.')
    st.stop()

last=valid.iloc[-1]
last_day=valid.index[-1].date()
st.info(f'최신 수신 일봉: {last_day} · 종가 {last.Close:,.0f} · 거래소의 종가 확정 여부는 별도 확인 필요')
if bool(last.A): label='A 매수 후보 (다음 거래일 확인)'
elif bool(last.B): label='B 매수 후보 (다음 거래일 확인)'
elif bool(last['exit']): label='기술적 청산 조건 (보유자만)'
elif bool(last.warning): label='주의 / 관망'
else: label='관망'
st.subheader(f'최신 기술적 판단: {label}')
st.caption('신호는 당일 종가로 계산됩니다. 실제 주문은 다음 거래일 시가 기준으로만 백테스트합니다. 장중 데이터는 최종 신호가 아닙니다.')

c1,c2,c3,c4=st.columns(4)
c1.metric('20일 이동평균',f'{last.MA20:,.0f}')
c2.metric('60일 이동평균',f'{last.MA60:,.0f}')
c3.metric('스토캐스틱 5 K/D',f'{last.K5:.1f} / {last.D5:.1f}')
c4.metric('스토캐스틱 20 K/D',f'{last.K20:.1f} / {last.D20:.1f}')

if view=='3개월': plot=valid.tail(65)
elif view=='6개월': plot=valid.tail(130)
elif view=='1년': plot=valid.tail(260)
else: plot=valid
fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.52,.16,.16,.16],subplot_titles=('가격 · 볼린저밴드 · 이동평균선 · 신호','Slow (5,3,3)','Slow (10,6,6)','Slow (20,12,12)'))
fig.add_trace(go.Candlestick(x=plot.index,open=plot.Open,high=plot.High,low=plot.Low,close=plot.Close,name='일봉'),row=1,col=1)
for col,name,color in [('BB_upper','BB 상단','#6b77a5'),('BB_lower','BB 하단','#6b77a5'),('MA5','MA5','#1e9c76'),('MA10','MA10','#e07c39'),('MA20','MA20','#9b59b6'),('MA60','MA60','#4c637a')]:
    fig.add_trace(go.Scatter(x=plot.index,y=plot[col],mode='lines',name=name,line=dict(width=1.3,color=color)),row=1,col=1)
for col,name,color,marker in [('A','A 매수','#159947','triangle-up'),('B','B 매수','#2386dd','triangle-up'),('exit','기술적 청산','#d13d3d','x')]:
    subset=plot[plot[col]]
    fig.add_trace(go.Scatter(x=subset.index,y=subset.Close,mode='markers',name=name,marker=dict(symbol=marker,size=11,color=color)),row=1,col=1)
for r,n in [(2,5),(3,10),(4,20)]:
    fig.add_trace(go.Scatter(x=plot.index,y=plot[f'K{n}'],name=f'K{n}',line=dict(color='#d34a4a',width=1.5)),row=r,col=1)
    fig.add_trace(go.Scatter(x=plot.index,y=plot[f'D{n}'],name=f'D{n}',line=dict(color='#32865d',width=1.5)),row=r,col=1)
    fig.add_hline(y=80,row=r,col=1,line_dash='dot',line_color='#aaaaaa')
    fig.add_hline(y=20,row=r,col=1,line_dash='dot',line_color='#aaaaaa')
    fig.update_yaxes(range=[0,100],row=r,col=1)
fig.update_layout(height=980,xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=-.07),margin=dict(l=5,r=5,t=55,b=40))
st.plotly_chart(fig,use_container_width=True)
st.caption('차트의 매도 표식은 기술적 조건 발생일이며, 실제 보유 중 실행된 매도 거래와는 다를 수 있습니다.')

st.header('A/B 전략별 백테스트')
results=[]
for strategy in ['A','B']:
    trades,equity=backtest(valid,strategy,fee,hold)
    metrics=summary(trades,equity)
    results.append({'전략':strategy,'총수익률 (%)':round(metrics['total'],2),'최대낙폭 (%)':round(metrics['mdd'],2),'완료 거래':metrics['count'],'승률 (%)':round(metrics['win'],1)})
    with st.expander(f'{strategy} 전략 거래 내역 ({len(trades)}건)'):
        if trades.empty: st.write('완료된 거래가 없습니다.')
        else: st.dataframe(trades.sort_values('exit_date',ascending=False),use_container_width=True)
st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True)
st.caption('각 전략을 독립적으로 100% 투자한 가상 성과. 종가 신호→다음 거래일 시가 체결, 편도 거래비용 반영. 마지막 미청산 포지션은 종가 평가하며 완료 거래 승률에는 포함하지 않습니다. 배당·시장충격·거래정지·시가 갭에 따른 실제 체결 위험은 완전히 재현하지 못합니다.')
st.warning('주의: 데이터 공급자의 수정주가(auto_adjust=True)를 사용합니다. 조정 OHLC와 실제 체결가격은 다를 수 있으며, 전략은 검증되지 않았습니다. 실거래 전 장기간/여러 종목의 표본외 검증이 필요합니다.')
