# GS칼텍스 국내영업 Network 분석 및 의사결정 지원 도구

공개자료 기반 포트폴리오용 Streamlit 앱입니다. 실제 GS칼텍스 내부 수익성/공급가격/물류비를 임의 추정하지 않습니다.

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 오피넷 API(선택)
`.streamlit/secrets.toml` 파일을 만들고 아래처럼 입력합니다.
```toml
OPINET_API_KEY = "발급받은_인증키"
```
API Key는 GitHub에 올리지 마세요.

## 포함 데이터
- `consumption_monthly.csv`: 2023-01~2026-07 지역별 실제 석유제품 소비량(천 Bbl)
- `vehicles_monthly.csv`: 2023-01~2026-07 자동차 등록대수(차종별)
- `network_snapshot.csv`: 고양·파주·김포·양주 오피넷 주유소 스냅샷
- `station_changes.csv`: 산업통상부 전국 주유소 등록/변동 이력
- `traffic_reference.csv`: 통계연보에서 대상 도시명이 명시된 교통량 조사 지점 참고자료(불완전 커버리지라 순위에는 미반영)

## 분석 원칙
- 실제 Network별 수익성이라고 표현하지 않음
- 공개되지 않은 판매량/마진/물류비/지원비/여신비용을 추정하지 않음
- 주유소 가격은 다운로드 시점의 스냅샷
- 교통량은 조사 지점 단위로만 해석
