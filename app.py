"""Swing Analysis 5.0 C/D — research dashboard, no order signals."""
import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from engine import VERSION, analyze_cd

st.set_page_config(page_title='Swing Analysis 5.0 C/D', page_icon='📈', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.block-container{max-width:1120px;padding-top:1rem;padding-bottom:3rem}
@media(max-width:680px){.block-container{padding-left:.65rem;padding-right:.65rem}}
h1{font-size:1.55rem!important;letter-spacing:-.04em}h2,h3{letter-spacing:-.03em}
[data-testid="stMetric"]{background:#f6f7f9;border-radius:13px;padding:12px 9px;border:1px solid #edf0f2}
[data-testid="stMetricLabel"]{font-size:.78rem!important}
[data-testid="stMetricValue"]{font-size:1.04rem!important}
[data-testid="stPlotlyChart"]{border:1px solid #e9ebef;border-radius:16px;overflow:hidden}
@media(prefers-color-scheme:dark){[data-testid="stMetric"]{background:#22272e;border-color:#303741}[data-testid="stPlotlyChart"]{border-color:#303741}}
</style>''', unsafe_allow_html=True)
st.title('📈 스윙 매매 분석 시스템 5.0 C/D')
st.caption('핵심 지지·저항 · 중복 관심 제거 · 모멘텀 파동 · 국면별 판단 근거 | 연구용, 매수·매도 주문 없음')

with st.expander('종목 및 차트 설정', expanded=True):
    symbol = st.text_input('종목 코드 (한국 .KS / .KQ, 미국 AAPL)', value='373220.KS').strip().upper()
    a,b = st.columns(2)
    years = a.selectbox('데이터 수집 기간', [3,5,7,10], index=2)
    view = b.selectbox('차트 범위', ['3개월','6개월','1년','전체'], index=1)
    show_bb = st.checkbox('볼린저밴드', value=False)
    show_ma = st.checkbox('20·60일 이동평균선', value=True)
    show_zones = st.checkbox('핵심 지지·저항 영역', value=True)
    show_watch = st.checkbox('대표 관심 표시 (매수 아님)', value=True)
    show_regime = st.checkbox('시장 국면 배경', value=True)
    st.caption('확정 정책: 20일선 이탈 3일 · 전고점 돌파 1% · 재지지 10일 · 최소 손익비 1.8 · ADX 참고값 18')

@st.cache_data(ttl=1800, show_spinner=False)
def fetch(ticker, years):
    end = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    raw = yf.download(ticker, start=str(end-dt.timedelta(days=years*366)), end=str(end), auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError('시세를 받지 못했습니다. 종목 코드 또는 네트워크를 확인하세요.')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw

upload = st.file_uploader('선택: 일봉 CSV 업로드 (Date,Open,High,Low,Close,Volume)', type=['csv'])
try:
    if upload is not None:
        raw = pd.read_csv(upload)
        raw.columns = [str(x).strip().capitalize() for x in raw.columns]
        if 'Date' not in raw.columns:
            raise ValueError('CSV에 Date 열이 필요합니다.')
        raw = raw.set_index(pd.to_datetime(raw['Date'])).drop(columns=['Date'])
    else:
        if not symbol:
            st.warning('종목 코드를 입력하세요.')
            st.stop()
        raw = fetch(symbol, years)
    d = analyze_cd(raw, ma_below_days=3, breakout_pct=.01, retest_days=10, min_rr=1.8, adx_ref=18)
except Exception as exc:
    st.error(f'데이터/분석 오류: {exc}')
    st.stop()

names = {'UP':'상승 추세','WEAK':'상승 약화','RANGE':'횡보','DOWN':'하락 추세','REVERSAL_PREP':'전환 준비','REVERSAL_CONFIRMED':'상승 전환 확인','NEUTRAL':'판별 유보','WARMUP':'초기 계산'}
colors = {'UP':'rgba(36,150,108,.07)','WEAK':'rgba(219,167,65,.09)','RANGE':'rgba(74,132,195,.07)','DOWN':'rgba(203,77,77,.08)','REVERSAL_PREP':'rgba(219,167,65,.08)','REVERSAL_CONFIRMED':'rgba(36,150,108,.11)','NEUTRAL':'rgba(125,125,125,.04)','WARMUP':'rgba(125,125,125,.02)'}
last = d.iloc[-1]
cols = st.columns(3)
cols[0].metric('시장 국면', names.get(last.regime, last.regime))
cols[1].metric('신규 매수', '금지' if last.buy_blocked else '진입 미검증')
cols[2].metric('모멘텀', '상승' if last.momentum_wave >= 0 else '하락')
st.caption(f'마지막 일봉 {d.index[-1]:%Y-%m-%d} · 종가 {last.Close:,.0f} · {last.reversal_step} · 실제 매수 신호가 아닙니다.')

n = {'3개월':65,'6개월':130,'1년':260,'전체':len(d)}[view]
p = d.tail(n).copy()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.77,.23], vertical_spacing=.035)
if show_regime:
    segments=[]
    for i, r in enumerate(p.regime):
        if not segments or segments[-1][2]!=r:
            segments.append([i,i,r])
        else:
            segments[-1][1]=i
    for a,b,r in segments:
        # date-based axes avoid mismatched category/numeric shape coordinates
        left = p.index[a]-pd.Timedelta(hours=12)
        right = p.index[b]+pd.Timedelta(hours=12)
        fig.add_vrect(x0=left,x1=right,fillcolor=colors.get(r,colors['NEUTRAL']),line_width=0,layer='below',row=1,col=1)
fig.add_trace(go.Candlestick(x=p.index,open=p.Open,high=p.High,low=p.Low,close=p.Close,
    name='일봉',increasing_line_color='#d05b5b',decreasing_line_color='#397fb9',showlegend=True),row=1,col=1)
if show_ma:
    for col,label,color,width in [('MA20','20일선','#9161ad',2),('MA60','60일선','#3973a4',2)]:
        fig.add_trace(go.Scatter(x=p.index,y=p[col],mode='lines',name=label,line=dict(color=color,width=width),hoverinfo='skip'),row=1,col=1)
if show_bb:
    for col,label in [('BB_upper','BB 상단'),('BB_lower','BB 하단')]:
        fig.add_trace(go.Scatter(x=p.index,y=p[col],mode='lines',name=label,line=dict(color='#9aaac1',width=1),hoverinfo='skip'),row=1,col=1)
if show_zones:
    # Draw only CURRENT confirmed structural zones, not future zones projected backwards.
    # Display from the most recent confirmation of the supporting pivot to the last bar.
    for lo,hi,label,color in [('support_zone_low','support_zone_high','현재 지지 영역','rgba(37,150,105,.12)'),('resistance_zone_low','resistance_zone_high','현재 저항 영역','rgba(203,91,91,.10)')]:
        if pd.notna(last[lo]) and pd.notna(last[hi]):
            # Current levels are intentionally horizontal guides, not historical backfilled signals.
            fig.add_hrect(y0=float(last[lo]),y1=float(last[hi]),fillcolor=color,line_width=0,row=1,col=1)
if show_watch:
    w = p[p.zone_watch]
    if not w.empty:
        fig.add_trace(go.Scatter(x=w.index,y=w.Low-w.ATR*.45,mode='markers',name='대표 관심 (매수 아님)',
            marker=dict(symbol='circle-open',size=10,color='#c69a3b',line=dict(width=2)),
            customdata=w.zone_watch_reason,hovertemplate='%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>'),row=1,col=1)
wave = p.momentum_wave.fillna(0)
fig.add_trace(go.Bar(x=p.index,y=wave,marker_color=np.where(wave>=0,'#32966d','#cb6464'),name='모멘텀 파동',hovertemplate='%{x|%Y-%m-%d}<br>파동 %{y:.3f}<extra></extra>'),row=2,col=1)
fig.add_hline(y=0,line_color='#9ca3af',line_width=1,row=2,col=1)
fig.update_layout(height=590,margin=dict(l=4,r=8,t=12,b=5),xaxis_rangeslider_visible=False,
    xaxis2=dict(tickformat='%y-%m-%d',nticks=5,showgrid=False),yaxis=dict(side='right',tickformat=',.0f',showgrid=True,gridcolor='rgba(128,128,128,.12)'),
    yaxis2=dict(side='right',showgrid=False,zeroline=False),legend=dict(orientation='h',y=-.10,font=dict(size=10)),
    hovermode='x unified',dragmode='pan',bargap=.16,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
st.subheader('① 주가 구조 + ② 모멘텀 파동')
st.plotly_chart(fig,use_container_width=True,key='swing50cd_chart',config={'displaylogo':False,'scrollZoom':True})
st.caption('녹색/빨강 영역은 현재 확정된 지지·저항 가격대의 수평 가이드입니다. 과거 시점에 이미 알려진 가격대였다는 뜻은 아닙니다. 빈 원은 구간별 첫 관심 이벤트이며 매수가 아닙니다.')

st.subheader('③ 매매 판단 근거')
if last.buy_blocked:
    st.error('🔒 신규 매수 금지 · '+last.block_reasons)
else:
    st.info('기본 매수 금지 조건 없음 · 실제 진입·손절·목표가와 손익비 1.8 검증 전이므로 매수 확정이 아닙니다.')
left,right=st.columns(2)
with left:
    st.markdown('**가격 구조**')
    st.write('지지:',last.support_levels or '확정된 근접 지지 없음')
    st.write('저항:',last.resistance_levels or '확정된 근접 저항 없음')
    st.write('지지 중요도:',int(last.support_strength),'/ 100 (연구용 상대점수)')
with right:
    st.markdown('**추세·모멘텀**')
    st.write('전환 단계:',last.reversal_step)
    st.write('MACD 파동:',last.momentum_state)
    st.write('RSI(14):',f'{last.RSI14:.1f}' if pd.notna(last.RSI14) else '계산 중')
    st.write('ADX(14):',f'{last.ADX:.1f}' if pd.notna(last.ADX) else '계산 중')

st.subheader('④ 날짜별 분석 기록')
summary=pd.DataFrame({
    '날짜':d.index.strftime('%Y-%m-%d'), '종가':d.Close.round(2).values,
    '국면':d.regime.map(names).values,'전환 단계':d.reversal_step.values,
    '신규 매수 금지':d.buy_blocked.map({True:'금지',False:'기본 차단 없음'}).values,
    '금지 사유':d.block_reasons.values,'대표 관심':d.zone_watch.map({True:'관심',False:''}).values,
    '관심 근거':d.zone_watch_reason.values,'지지 중요도':d.support_strength.values,
    '모멘텀 파동':d.momentum_wave.round(3).values,'지지 가격대':d.support_levels.values,
    '저항 가격대':d.resistance_levels.values})
st.dataframe(summary.tail(n).iloc[::-1],hide_index=True,use_container_width=True,height=330)
st.download_button('전체 분석 기록 CSV 다운로드',summary.to_csv(index=False).encode('utf-8-sig'),
    file_name='swing_5_0_cd_diagnostics.csv',mime='text/csv',key='swing50cd_download')
a,b,c=st.columns(3)
a.metric('표시 구간 일봉',f'{len(p):,}')
b.metric('대표 관심 이벤트',int(p.zone_watch.sum()))
c.metric('하락 국면 차단 위반',int(((d.regime=='DOWN')&(~d.buy_blocked)).sum()))
with st.expander('현재 버전의 범위와 제한'):
    st.markdown('''- **5.0-C/D는 진단 전용**입니다. 실제 매수·매도·백테스트 체결은 구현하지 않았습니다.
- 지지·저항은 확인 완료된 일봉 피벗으로 계산합니다. 주봉 피벗·피보나치·엘리어트 파동 자동 카운팅은 아직 미구현입니다.
- 모멘텀은 MACD(12,26,9) 히스토그램을 ATR로 나눈 연구용 시각화이며 단독 매수 근거가 아닙니다.
- 관심 이벤트는 확정 피벗 그룹별 첫 접근만 표시합니다. 실제 지지선 강도와 신호 품질은 여러 종목에서 검증해야 합니다.
- 손익비 1.8은 향후 실제 진입·손절·목표가를 산출한 다음 적용합니다. 기본 금지 해제는 매수 확정이 아닙니다.
- 지지/저항 색상 영역은 **현재 시점** 가이드이며 과거 전체 기간에 소급 표시되는 시각적 보조 요소입니다. 과거 당시 수준은 CSV의 날짜별 값을 확인하세요.''')
