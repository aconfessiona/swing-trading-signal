# 스윙 매매 4.7.2 — 버전 불일치 방지 수정

- `app.py`에 `prepare()`와 `simulate()` 분석 엔진을 내장했습니다.
- 따라서 GitHub 또는 Streamlit Cloud에 이전 `engine.py`가 남아 있어도 앱이 잘못된 함수를 가져오는 문제가 발생하지 않습니다.
- `engine.py`도 동일한 코드로 함께 제공하며, 추후 모듈화할 때 사용합니다.
- 매수·매도 전략은 4.7.1과 동일합니다. 최종 청산 마커는 숨기되 백테스트에는 반영합니다.
- 실제 Streamlit Cloud 배포 테스트는 아직 하지 않았습니다.

## GitHub 업로드
`app.py`, `engine.py`, `requirements.txt`, `README.md`

Commit: `Fix engine version mismatch v4.7.2`
