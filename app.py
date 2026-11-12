"""Mobile-first swing trading 4.0; research only, no orders."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from engine import version4, backtest_v4

st.set_page_config(page_title='스윙 매매 신호 4.0',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding-top:1rem;max-width:1300px}@media(max-width:700px){.block-container{padding-left:.6rem;padding-right:.6rem}h1{font-size:1.7rem!important}}</style>''',unsafe_allow_html=True)
st.title('📈 스윙 매매 신호 4.0')
st.caption('1~4주 스윙 · 기존 A/B 매수 · 저항에서 50% 분할 매도 · 추세 이탈 시 잔여 청산 · 연구용')
with st.expander('⚙️ 종목·설정',expanded=True):
    ticker=st.text_input('종목 코드 (LG엔솔: 373220.KS)',value='373220.KS').strip().upper()
    c1,c2=st.columns(2)
    years=c1.selectbox('조회 기간(년)',[2,3,5],index=1)
    view=c2.selectbox('차트 기간',['3개월','6개월','1년','전체'],index=1)
    c3,c4=st.columns(2)
    fee=c3.number_input('편도 거래비용(bp)',0,200,15,5)
    max_hold=c4.slider('최대 보유 거래일',10,40,20)
    atr_multiple=st.slider('ATR 보호 청산 배수',1.5,4.0,2.5,0.25)
    st.caption('최대 보유기간은 거래일 기준입니다. 20거래일≈4주.')
@st.cache_data(ttl=1800,show_spinner=False)
def load(symbol,yrs):
    end=dt.datetime.now(dt.timezone.utc).date()+dt.timedelta(days=1)
    start=end-dt.timedelta(days=366*yrs)
    raw=yf.download(symbol,start=str(start),end=str(end),auto_adjust=True,progress=False,threads=False)
    if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
    if raw.empty:raise ValueError('가격 데이터가 없습니다. 종목코드를 확인하세요.')
    raw.index=pd.to_datetime(raw.index)
    if raw.index.tz is not None:raw.index=raw.index.tz_localize(None)
    return raw
uploaded=st.file_uploader('선택: 일봉 CSV 업로드 (Date,Open,High,Low,Close,Volume)',type=['csv'])
try:
    if uploaded is not None:
        raw=pd.read_csv(uploaded)
        raw.columns=[str(c).strip().capitalize() for c in raw.columns]
        if 'Date' not in raw.columns:raise ValueError('CSV에 Date 열이 필요합니다.')
        raw['Date']=pd.to_datetime(raw['Date']);raw=raw.set_index('Date').sort_index()
    else:raw=load(ticker,years)
    d=version4(raw).dropna(subset=['MA60','K20','D20']).copy()
    if len(d)<85:raise ValueError('지표 계산에 필요한 과거 데이터가 부족합니다.')
except Exception as e:
    st.error(f'데이터/지표 오류: {e}');st.stop()
last=d.iloc[-1]
st.info(f'최신 수신 일봉: {d.index[-1]:%Y-%m-%d} · 종가 {last.Close:,.0f} · 장중 데이터는 확정 신호가 아닙니다.')
st.caption(f'RSI(14) {last.RSI14:.1f} · ATR(14) {last.ATR14:,.0f} · 저항 접근 {"예" if last.resistance_near else "아니오"} · 1차 매도 조건 {"충족" if last.sell_half_condition else "미충족"}')

aev,aeq,aopen=backtest_v4(d,'A',fee,max_hold,atr_multiple)
bev,beq,bopen=backtest_v4(d,'B',fee,max_hold,atr_multiple)
strategy=st.radio('차트에 표시할 포지션별 신호',['A','B'],horizontal=True)
ev= aev if strategy=='A' else bev
n={'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[view]
p=d.tail(n).copy(); x=p.index.strftime('%Y-%m-%d').tolist()
st.subheader('① 일봉 · 저항선 · 실제 백테스트 매매')
with st.expander('표시 설정',expanded=False):
    lines=st.multiselect('표시할 선',['볼린저밴드','20일선','60일선','상승 채널','과거 60일 고점','주봉 저항','거래량 집중 가격대'],default=['볼린저밴드','20일선','60일선','상승 채널','과거 60일 고점','주봉 저항'])
fig=go.Figure()
fig.add_trace(go.Candlestick(x=x,open=p.Open,high=p.High,low=p.Low,close=p.Close,name='일봉',increasing_line_color='#d84a4a',decreasing_line_color='#2879cb',increasing_fillcolor='#d84a4a',decreasing_fillcolor='#2879cb'))
def line(col,name,color,width=1.5,dash='solid'):
    fig.add_trace(go.Scatter(x=x,y=p[col],mode='lines',name=name,line=dict(color=color,width=width,dash=dash)))
if '볼린저밴드' in lines:
    line('BB_upper','BB 상단','#8292b9');line('BB_lower','BB 하단','#8292b9')
if '20일선' in lines:line('MA20','20일선','#9852b2',2.2)
if '60일선' in lines:line('MA60','60일선','#30669d',2.2)
if '상승 채널' in lines:
    line('channel_upper','채널 상단','#d29a31',1.5,'dot');line('channel_lower','채널 하단','#169b76',1.5,'dot')
if '과거 60일 고점' in lines:line('prior_high_60','이전 고점','#dc8043',1.3,'dash')
if '주봉 저항' in lines:
    line('week_prior_high','주봉 이전 고점','#a66c5b',1.5,'dash');line('week_upper','주봉 BB 상단','#bb8a42',1.4,'dot')
if '거래량 집중 가격대' in lines:line('volume_resistance','거래량 집중 근사','#666666',1.5,'dash')
if not ev.empty:
    inview=ev[ev['signal_date'].isin(p.index)]
    for action,name,color,symbol in [('매수','매수','#14985c','triangle-up'),('1차 매도','1차 50% 매도','#e5a11c','circle'),('전량 청산','잔여/보호 청산','#d33e45','x')]:
        selected=inview[inview.action==action]
        if selected.empty:continue
        dates=selected.signal_date.dt.strftime('%Y-%m-%d').tolist()
        ys=[float(p.loc[date,'Low']*.975 if action=='매수' else p.loc[date,'High']*1.025) for date in selected.signal_date]
        fig.add_trace(go.Scatter(x=dates,y=ys,mode='markers',name=name,marker=dict(symbol=symbol,color=color,size=13,line=dict(color='white',width=1)),customdata=selected['reason'],hovertemplate='%{x}<br>'+name+'<br>%{customdata}<extra></extra>'))
fig.update_layout(height=570,margin=dict(l=0,r=0,t=12,b=20),xaxis=dict(type='category',nticks=5,rangeslider_visible=False),yaxis=dict(side='right',tickformat=',.0f'),legend=dict(orientation='h',y=-.13,font=dict(size=10)),hovermode='x unified',dragmode='pan',uirevision=f'{ticker}-{view}-{strategy}')
st.plotly_chart(fig,key='v4_price',use_container_width=True,config={'displaylogo':False,'scrollZoom':True,'responsive':True})
st.caption('마커는 신호 발생일 종가 기준, 실제 백테스트 체결은 다음 거래일 시가 기준입니다. 주봉 값은 완료된 주만 사용합니다.')

st.subheader('② 세 가지 스토캐스틱')
tabs=st.tabs(['5·3·3','10·6·6','20·12·12'])
for tab,n in zip(tabs,[5,10,20]):
    with tab:
        f=go.Figure()
        f.add_trace(go.Scatter(x=x,y=p[f'K{n}'],name='%K',line=dict(color='#d94b4b',width=2)))
        f.add_trace(go.Scatter(x=x,y=p[f'D{n}'],name='%D',line=dict(color='#169b76',width=2)))
        for val in (20,80):f.add_hline(y=val,line_color='#888',line_dash='dot')
        f.update_layout(height=320,margin=dict(l=0,r=0,t=15,b=10),xaxis=dict(type='category',nticks=5),yaxis=dict(range=[0,100],side='right'),legend=dict(orientation='h',y=1.15),dragmode='pan')
        st.plotly_chart(f,key=f'v4_stoch_{n}',use_container_width=True,config={'displaylogo':False,'scrollZoom':True})
        st.caption(f'최근 %K {last[f"K{n}"]:.1f} · %D {last[f"D{n}"]:.1f}')

st.subheader('③ 매매 기록')
if ev.empty:st.write('완료된 매매 신호가 없습니다.')
else:
    table=ev.copy();table['signal_date']=table.signal_date.dt.strftime('%Y-%m-%d');table['date']=table.date.dt.strftime('%Y-%m-%d');table['price']=table.price.round(2)
    st.dataframe(table.iloc[::-1],hide_index=True,use_container_width=True)
st.subheader('④ 전략 A/B 백테스트')
rows=[]
for name,equity,events,opened in [('A',aeq,aev,aopen),('B',beq,bev,bopen)]:
    dd=(equity/equity.cummax()-1).min()*100 if len(equity) else 0
    rows.append({'전략':name,'총수익률(%)':round((equity.iloc[-1]-1)*100,2),'최대낙폭(%)':round(dd,2),'매수 횟수':int((events.action=='매수').sum()) if not events.empty else 0,'1차 매도 횟수':int((events.action=='1차 매도').sum()) if not events.empty else 0,'청산 횟수':int((events.action=='전량 청산').sum()) if not events.empty else 0,'미청산 보유':opened})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.warning('백테스트는 연구용입니다. 과거 성과는 미래 수익을 보장하지 않습니다. 주봉 확정, 거래비용, 갭 위험, 슬리피지, 미체결, 데이터 품질을 확인하세요. 실제 주문 기능은 없습니다.')
