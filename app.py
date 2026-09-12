from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================================================
# Page / constants
# =========================================================
st.set_page_config(
    page_title="GS칼텍스 서울 Network 분석",
    page_icon="⛽",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TARGET_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
]
FUEL_COLS = {
    "휘발유": ("gasoline_kbbl", "gasoline_price", "B027"),
    "경유": ("diesel_kbbl", "diesel_price", "D047"),
}
BRAND_CODE_NAME = {
    "GSC": "GS칼텍스", "SKE": "SK에너지", "SOL": "S-OIL", "HDO": "HD현대오일뱅크",
    "RTE": "알뜰주유소", "RTX": "고속도로알뜰", "NHO": "NH-OIL", "ETC": "자가상표",
    "E1G": "E1", "SKG": "SK가스",
}
SNAPSHOT_DATE = "2026.09.12"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.18); padding: 12px 14px; border-radius: 12px;}
      .small-note {font-size: .88rem; opacity: .78;}
      .guide-box {padding: 1rem 1.1rem; border-radius: .7rem; background: rgba(120,120,120,.08); margin-bottom: 1rem; line-height:1.7;}
      .conclusion-box {padding: 1rem 1.1rem; border-left: 4px solid rgba(80,120,180,.85); background: rgba(120,120,120,.055); border-radius: .35rem; margin-top: 1rem; line-height:1.75;}
      .section-text {line-height:1.85; font-size:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Data
# =========================================================
@st.cache_data
def load_data():
    cons = pd.read_csv(DATA_DIR / "consumption_monthly.csv")
    veh = pd.read_csv(DATA_DIR / "vehicles_monthly.csv")
    net = pd.read_csv(DATA_DIR / "network_snapshot.csv")

    cons["month"] = pd.to_datetime(cons["month"] + "-01")
    veh["month"] = pd.to_datetime(veh["month"] + "-01")

    for c in ["gasoline_price", "diesel_price", "premium_gasoline_price", "kerosene_price"]:
        if c in net.columns:
            net[c] = pd.to_numeric(net[c], errors="coerce")
    return cons, veh, net


cons, veh, net = load_data()

# =========================================================
# Helpers
# =========================================================
def district_vehicle_series(district: str) -> pd.DataFrame:
    d = veh[(veh["sido"] == "서울") & (veh["sigungu"] == district)].copy()
    if d.empty:
        return pd.DataFrame(columns=["month", "count"])

    if "vehicle_type" in d.columns and (d["vehicle_type"] == "총계").any():
        d = d[d["vehicle_type"] == "총계"]
    d = d.groupby("month", as_index=False)["count"].sum()
    return d.sort_values("month")


def latest_vehicle(district: str):
    d = district_vehicle_series(district)
    if d.empty:
        return np.nan, np.nan
    latest = d.iloc[-1]
    latest_n = float(latest["count"])
    prev = d[d["month"] == latest["month"] - pd.DateOffset(years=1)]
    yoy = np.nan if prev.empty or float(prev.iloc[0]["count"]) == 0 else (latest_n / float(prev.iloc[0]["count"]) - 1) * 100
    return latest_n, yoy


def safe_rank_pct(s: pd.Series, ascending=True):
    return s.rank(pct=True, ascending=ascending, method="average") * 100


def network_summary() -> pd.DataFrame:
    rows = []
    for district in TARGET_DISTRICTS:
        d = net[net["district"] == district].copy()
        total = len(d)
        gs = int((d["brand"] == "GS칼텍스").sum())
        vehicles, vehicle_yoy = latest_vehicle(district)
        rows.append({
            "district": district,
            "stations": total,
            "gs_stations": gs,
            "gs_share": gs / total * 100 if total else np.nan,
            "vehicles": vehicles,
            "vehicle_yoy": vehicle_yoy,
            "vehicles_per_station": vehicles / total if total and pd.notna(vehicles) else np.nan,
            "gs_gasoline": d.loc[d["brand"] == "GS칼텍스", "gasoline_price"].mean(),
            "district_gasoline": d["gasoline_price"].mean(),
            "gs_diesel": d.loc[d["brand"] == "GS칼텍스", "diesel_price"].mean(),
            "district_diesel": d["diesel_price"].mean(),
        })
    out = pd.DataFrame(rows)
    out["gs_gasoline_gap"] = out["gs_gasoline"] - out["district_gasoline"]
    out["gs_diesel_gap"] = out["gs_diesel"] - out["district_diesel"]
    out["avg_price_gap"] = out[["gs_gasoline_gap", "gs_diesel_gap"]].mean(axis=1)

    # 상대 비교용 스크리닝 점수. 실제 수익성 점수가 아님.
    share_low = safe_rank_pct(out["gs_share"], ascending=False)
    vps_high = safe_rank_pct(out["vehicles_per_station"], ascending=True)
    yoy_fill = out["vehicle_yoy"].fillna(out["vehicle_yoy"].median())
    yoy_high = safe_rank_pct(yoy_fill, ascending=True)
    out["expansion_score"] = (share_low + vps_high + yoy_high) / 3

    share_high = safe_rank_pct(out["gs_share"], ascending=True)
    vps_low = safe_rank_pct(out["vehicles_per_station"], ascending=False)
    out["efficiency_score"] = (share_high + vps_low) / 2

    gap_fill = out["avg_price_gap"].fillna(out["avg_price_gap"].median())
    out["price_watch_score"] = safe_rank_pct(gap_fill, ascending=True)

    med_share = out["gs_share"].median()
    med_vps = out["vehicles_per_station"].median()

    def label(r):
        if r["gs_share"] < med_share and r["vehicles_per_station"] > med_vps:
            return "Network 확대 우선검토"
        if r["gs_share"] >= med_share and r["vehicles_per_station"] <= med_vps:
            return "기존 Network 효율화 우선검토"
        return "개별 Network·가격 경쟁 추가점검"

    out["primary_action"] = out.apply(label, axis=1)
    return out


summary = network_summary()


def get_api_key():
    try:
        return st.secrets.get("OPINET_API_KEY", "")
    except Exception:
        return ""


def opinet_around(key: str, x: float, y: float, radius: int, prodcd: str):
    url = "https://www.opinet.co.kr/api/aroundAll.do"
    params = {"certkey": key, "out": "json", "x": x, "y": y, "radius": radius, "prodcd": prodcd, "sort": 2}
    r = requests.get(url, params=params, timeout=7)
    r.raise_for_status()
    js = r.json()
    result = js.get("RESULT")
    if not isinstance(result, dict) or "OIL" not in result:
        raise ValueError("오피넷 응답 형식이 예상과 다릅니다.")
    oils = result.get("OIL", [])
    if isinstance(oils, dict):
        oils = [oils]
    df = pd.DataFrame(oils)
    if not df.empty:
        code_series = df["POLL_DIV_CD"] if "POLL_DIV_CD" in df.columns else pd.Series("", index=df.index)
        df["브랜드"] = code_series.map(BRAND_CODE_NAME).fillna(code_series)
        for c in ["PRICE", "DISTANCE"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def guide(title: str, why: str, how: str, conclusion: str):
    st.markdown(
        f"""
        <div class="guide-box">
        <b>{title}</b><br><br>
        <b>왜 보나요?</b> {why}<br><br>
        <b>어떻게 보나요?</b> {how}<br><br>
        <b>무엇을 판단하나요?</b> {conclusion}
        </div>
        """,
        unsafe_allow_html=True,
    )


def conclusion_box(lines):
    st.markdown("### 이 탭에서 얻을 수 있는 결론은 무엇인가요?")
    body = "<br>".join(lines)
    st.markdown(f'<div class="conclusion-box">{body}</div>', unsafe_allow_html=True)


def fmt_num(x, digits=0):
    if pd.isna(x):
        return "확인 불가"
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def price_position(percentile):
    if pd.isna(percentile):
        return "가격 위치 확인 불가"
    p = float(percentile)
    if p >= 50:
        top = max(1, round(100 - p))
        if p >= 80:
            tone = "가격이 높은 편"
        elif p >= 60:
            tone = "가격이 다소 높은 편"
        else:
            tone = "중간보다 약간 높은 편"
        return f"백분위 {p:.0f}% · 자치구 내 가격 상위 약 {top}% · {tone}"
    bottom = max(1, round(p))
    if p <= 20:
        tone = "가격이 낮은 편"
    elif p <= 40:
        tone = "가격이 다소 낮은 편"
    else:
        tone = "중간보다 약간 낮은 편"
    return f"백분위 {p:.0f}% · 자치구 내 가격 하위 약 {bottom}% · {tone}"


# =========================================================
# Header
# =========================================================
st.title("⛽ GS칼텍스 서울 주유소 Network 분석 및 의사결정 지원 도구")
st.caption(f"서울특별시 25개 자치구 · 오피넷 Network/가격 기준일 {SNAPSHOT_DATE} · 공개자료 기반")

with st.expander("분석 원칙과 공개자료의 한계", expanded=False):
    st.markdown(
        """
- 목적: 서울 전체 석유제품 수요와 25개 자치구의 차량시장·주유소 Network·가격을 연결해 영업 담당자가 추가 검토할 지역을 찾습니다.
- 실제 수익성 계산 아님: 주유소별 판매량, GS칼텍스 공급가격, 물류비, 판촉·지원비, 여신·채권, 계약조건이 없으므로 실제 이익을 추정하지 않습니다.
- 분석 단위: 서울 전체 수요 → 자치구 시장 → 자치구 Network → 가격 경쟁 → 개별 GS Network로 범위를 좁힙니다.
- 가격/Network 데이터: 오피넷에서 내려받은 2026년 9월 12일 기준 스냅샷입니다. 가격의 장기 추세가 아니라 분석시점의 경쟁 포지션을 봅니다.
- 최종 점수: 절대적인 출점/철수 결론이 아니라 25개 구를 상대 비교해 추가 검토 순서를 만드는 스크리닝 지표입니다.
        """
    )

# =========================================================
# Tabs
# =========================================================
t1, t2, t3, t4, t5, t6 = st.tabs([
    "① 분석 개요", "② 서울 수요·시장", "③ Network 분석", "④ 가격·경쟁", "⑤ 개별 Network 진단", "⑥ 최종결론",
])

# ---------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------
with t1:
    st.subheader("분석의 출발점")
    st.markdown(
        """
        <div class="section-text">
        이 프로그램은 서울 25개 자치구 가운데 GS칼텍스가 신규 Network 확대를 먼저 검토할 지역과, 이미 확보한 Network의 효율성을 먼저 점검할 지역을 찾기 위해 만들었습니다. 단순히 주유소 수가 많은지 적은지만 비교하지 않고, 시장의 크기와 Network 침투 수준, 가격 경쟁 위치를 순서대로 연결해 영업 검토 우선순위를 좁히는 것이 핵심입니다.<br><br>
        분석은 서울 전체의 휘발유·경유 소비 흐름에서 시작합니다. 서울의 정유제품 시장이 어떤 방향으로 움직이는지를 확인한 뒤, 자치구별 자동차 등록대수와 전체 주유소 수를 이용해 지역별 잠재 시장규모와 Network 밀도를 비교합니다. 자치구별 석유 소비량은 공개자료에서 직접 확인하기 어려우므로 자동차 등록대수와 주유소당 등록차량은 실제 판매량이 아니라 시장규모를 가늠하기 위한 보조지표로 사용합니다.<br><br>
        다음 단계에서는 각 자치구에서 GS칼텍스 주유소가 전체 주유소 중 어느 정도의 비중을 차지하는지 확인합니다. 차량시장이 상대적으로 크지만 GS칼텍스 Network 비중이 낮다면 Network 확대 가능성을 추가로 검토할 수 있고, 반대로 GS칼텍스 비중이 이미 높은데 주유소당 잠재 차량시장이 상대적으로 작다면 신규 확대보다 현재 Network의 판매성과와 운영효율을 먼저 점검할 필요가 있습니다.<br><br>
        이후 가격 분석에서는 각 자치구의 평균가격과 GS칼텍스 평균가격, 경쟁 브랜드 평균가격을 비교합니다. 가격이 높다는 사실을 곧바로 문제로 보거나 가격인하 필요성으로 연결하지 않고, 고객이 받아들이는 가격 수준인지, 입지·서비스·프로모션과 함께 설명되는지, 인근 경쟁점 대비 어떤 위치인지 추가 확인해야 할 지역을 찾는 데 사용합니다.<br><br>
        마지막에는 자치구 단위 분석에서 개별 GS칼텍스 주유소까지 범위를 좁힙니다. 선택한 주유소가 자치구 전체 가격분포에서 상위 또는 하위 어느 수준에 있는지를 확인하고, 필요할 경우 오피넷 API의 최신 주변 경쟁정보를 참고합니다. 이렇게 시장 수요 → 자치구 시장 → Network → 가격 경쟁 → 개별 주유소의 순서로 분석한 뒤, Network 확대·기존 Network 효율화·가격전략 점검 중 어떤 검토를 먼저 할지를 제시합니다.<br><br>
        다만 이 프로그램의 결론은 출점, 철수, 가격변경을 확정하는 결론이 아닙니다. 공개자료만으로는 주유소별 판매량, 공급가격, 물류비, 판촉·시설지원비, 여신·채권, 계약조건, 신규 투자비 등을 확인할 수 없기 때문입니다. 따라서 본 프로그램은 실제 국내영업 의사결정에 앞서 어디부터 더 깊게 확인해야 하는지를 구조화하는 1차 스크리닝 도구로 해석합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    seoul_cons = cons[cons["region"] == "서울"]
    c1.metric("석유소비 관측기간", f"{seoul_cons['month'].min():%Y.%m} ~ {seoul_cons['month'].max():%Y.%m}")
    c2.metric("자동차 관측기간", f"{veh['month'].min():%Y.%m} ~ {veh['month'].max():%Y.%m}")
    c3.metric("분석 자치구", "25개 구")
    c4.metric("서울 주유소", f"{len(net):,}개")

    st.markdown("### 분석 흐름")
    flow = pd.DataFrame({
        "단계": ["1. 서울 수요", "2. 자치구 시장", "3. Network", "4. 가격 경쟁", "5. 개별 Network", "6. 영업 판단"],
        "핵심 질문": [
            "서울의 휘발유·경유 수요는 어떻게 움직이는가?",
            "25개 구의 차량시장 규모와 주유소 밀도는 어떻게 다른가?",
            "각 구에서 GS칼텍스 Network는 얼마나 확보되어 있는가?",
            "GS 가격은 해당 구와 경쟁사 대비 어느 위치인가?",
            "선택한 GS 주유소는 구 내 가격분포에서 어디에 위치하는가?",
            "확대·효율화·가격점검 중 무엇을 먼저 검토해야 하는가?",
        ],
        "사용 데이터": [
            "서울 휘발유·경유 월별 소비량",
            "자치구별 자동차 등록대수 + 주유소 수",
            "브랜드별 주유소 수",
            "브랜드별 휘발유·경유 가격",
            "개별 주유소 가격·주소",
            "앞선 지표의 상대비교 결과",
        ],
    })
    st.dataframe(flow, hide_index=True, use_container_width=True)

    st.markdown("### 이 분석이 도달하려는 결론")
    st.info(
        "시장규모에 비해 GS칼텍스 Network 비중이 낮은 자치구는 Network 확대 검토 후보로, GS Network가 상대적으로 높은데 주유소당 잠재 차량시장이 작은 곳은 기존 Network 효율화 검토 후보로 봅니다. 또한 GS칼텍스 가격이 자치구 평균보다 상대적으로 높은 곳은 가격전략을 추가 점검할 후보로 분류합니다. 이후 실제 의사결정에서는 내부 판매량·공급가격·물류비·판촉비·여신·채권·계약조건을 연결해 수익성을 검증해야 합니다."
    )

# ---------------------------------------------------------
# 2. Demand / market
# ---------------------------------------------------------
with t2:
    st.subheader("서울 석유제품 수요와 자치구 시장규모")
    guide(
        "① 수요에서 시작합니다.",
        "Network 전략을 보기 전에 서울의 정유제품 시장 자체가 어떻게 움직이는지 확인해야 지역 Network 확대 판단이 단순 점포 수 비교에 그치지 않습니다.",
        "서울 전체 휘발유·경유 월별 소비량을 확인한 뒤, 25개 구의 자동차 등록대수와 주유소당 등록차량을 비교합니다. 자치구별 석유 소비량은 공개자료에 없어 자동차 등록대수를 잠재 시장규모의 보조지표로 사용합니다.",
        "서울 전체 수요의 방향과 함께, 어떤 구가 상대적으로 큰 차량시장과 높은 Network 수용여력을 갖는지 확인합니다.",
    )

    fuel = st.radio("제품", ["휘발유", "경유"], horizontal=True, key="demand_fuel")
    volume_col, _, _ = FUEL_COLS[fuel]
    d = cons[cons["region"] == "서울"].sort_values("month").copy()

    latest_month = d["month"].max()
    latest_year = latest_month.year
    m = latest_month.month
    ytd = d[(d["month"].dt.year == latest_year) & (d["month"].dt.month <= m)][volume_col].sum()
    prev_ytd = d[(d["month"].dt.year == latest_year - 1) & (d["month"].dt.month <= m)][volume_col].sum()
    yoy = (ytd / prev_ytd - 1) * 100 if prev_ytd else np.nan

    a, b, c = st.columns(3)
    a.metric(f"{latest_year}년 1~{m}월 서울 {fuel}", f"{ytd:,.0f} 천Bbl")
    b.metric("전년 동기 대비", "-" if pd.isna(yoy) else f"{yoy:+.1f}%")
    b_latest = d.iloc[-1][volume_col]
    c.metric(f"{latest_month:%Y.%m} 월간 소비", f"{b_latest:,.0f} 천Bbl")

    fig = px.line(d, x="month", y=volume_col, markers=True, labels={"month": "월", volume_col: "천 Bbl"}, title=f"서울 {fuel} 월별 소비량")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

    market = summary[["district", "vehicles", "vehicle_yoy", "stations", "vehicles_per_station"]].sort_values("vehicles_per_station", ascending=False).copy()
    left, right = st.columns([1.15, 1])
    with left:
        fig2 = px.bar(market, x="district", y="vehicles_per_station", labels={"district": "자치구", "vehicles_per_station": "주유소당 등록차량"}, title="자치구별 주유소당 등록차량")
        fig2.update_layout(height=450, xaxis_tickangle=-45, margin=dict(l=10, r=10, t=55, b=80))
        st.plotly_chart(fig2, use_container_width=True)
    with right:
        display = market.copy()
        display.columns = ["자치구", "자동차 등록대수", "자동차 YoY(%)", "전체 주유소", "주유소당 등록차량"]
        st.dataframe(
            display.style.format({"자동차 등록대수": "{:,.0f}", "자동차 YoY(%)": "{:+.2f}", "전체 주유소": "{:,.0f}", "주유소당 등록차량": "{:,.0f}"}),
            hide_index=True, use_container_width=True, height=450,
        )
    st.caption("자동차 등록대수와 주유소당 등록차량은 실제 유류 판매량이 아니라 자치구별 잠재 시장규모와 Network 밀도를 비교하기 위한 보조지표입니다.")

    district_d = st.selectbox("결론을 자세히 볼 자치구", TARGET_DISTRICTS, key="demand_conclusion_district")
    rr = summary[summary["district"] == district_d].iloc[0]
    avg_vps = summary["vehicles_per_station"].mean()
    vps_side = "서울 평균보다 높은" if rr["vehicles_per_station"] > avg_vps else "서울 평균보다 낮은"
    yoy_text = "증가" if pd.notna(rr["vehicle_yoy"]) and rr["vehicle_yoy"] > 0 else "감소"
    conclusion_box([
        f"서울 {fuel} 수요는 {latest_year}년 1~{m}월 누적으로 전년 동기 대비 {yoy:+.1f}% 변했습니다. 이는 서울 전체 정유제품 시장의 방향을 확인하는 기준입니다.",
        f"{district_d}의 자동차 등록대수는 {fmt_num(rr['vehicles'])}대이며, 주유소 1곳당 등록차량은 {fmt_num(rr['vehicles_per_station'])}대로 {vps_side} 수준입니다.",
        f"자동차 등록대수는 전년 동월 대비 {rr['vehicle_yoy']:+.2f}%로 {yoy_text}해 지역 차량시장의 방향도 함께 확인할 수 있습니다.",
        "따라서 이 탭에서는 서울 전체 수요와 자치구별 잠재 시장규모를 먼저 확인하고, 다음 Network 탭에서 GS칼텍스 침투 수준과 연결해 확대 여부를 검토합니다.",
    ])

# ---------------------------------------------------------
# 3. Network
# ---------------------------------------------------------
with t3:
    st.subheader("서울 25개 자치구 Network 구조")
    guide(
        "② 시장규모 다음에는 Network 침투도를 봅니다.",
        "시장규모가 커도 GS칼텍스 Network가 이미 충분히 확보된 곳과 상대적으로 부족한 곳은 영업전략이 달라야 합니다.",
        "자치구별 전체 주유소와 브랜드별 주유소 수, GS칼텍스 Network 비중을 비교하고 앞 탭의 주유소당 등록차량과 함께 봅니다.",
        "시장규모 대비 GS Network가 상대적으로 부족한 구와 이미 높은 비중을 확보한 구를 구분해 확대와 효율화 후보군을 만듭니다.",
    )

    cnt = net.groupby(["district", "brand"]).size().reset_index(name="주유소수")
    fig = px.bar(cnt, x="district", y="주유소수", color="brand", barmode="stack", title="서울 자치구별 브랜드 Network 구성")
    fig.update_layout(height=470, xaxis_title="", xaxis_tickangle=-45, legend_title="상표", margin=dict(l=10, r=10, t=55, b=85))
    st.plotly_chart(fig, use_container_width=True)

    show = summary[["district", "stations", "gs_stations", "gs_share", "vehicles_per_station"]].sort_values("gs_share", ascending=False).copy()
    show.columns = ["자치구", "전체 주유소", "GS칼텍스", "GS Network 비중(%)", "주유소당 등록차량"]
    st.dataframe(
        show.style.format({"전체 주유소": "{:,.0f}", "GS칼텍스": "{:,.0f}", "GS Network 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}"}),
        hide_index=True, use_container_width=True, height=490,
    )

    district_n = st.selectbox("결론을 자세히 볼 자치구", TARGET_DISTRICTS, key="network_district")
    r = summary[summary["district"] == district_n].iloc[0]
    med_share = summary["gs_share"].median()
    med_vps = summary["vehicles_per_station"].median()
    share_desc = "중앙값보다 낮아" if r["gs_share"] < med_share else "중앙값보다 높아"
    vps_desc = "중앙값보다 높습니다" if r["vehicles_per_station"] > med_vps else "중앙값보다 낮습니다"
    if r["gs_share"] < med_share and r["vehicles_per_station"] > med_vps:
        action_text = "시장규모 대비 GS Network 침투가 상대적으로 낮아 Network 확대 가능성을 우선 검토할 후보로 볼 수 있습니다."
    elif r["gs_share"] >= med_share and r["vehicles_per_station"] <= med_vps:
        action_text = "GS Network는 상대적으로 충분한 반면 주유소당 잠재 차량시장은 작아 신규 확대보다 기존 Network 효율을 먼저 점검할 후보로 볼 수 있습니다."
    else:
        action_text = "Network 비중과 시장규모가 한 방향으로 뚜렷하지 않아 가격과 개별 주유소 경쟁환경을 함께 확인하는 것이 적절합니다."
    conclusion_box([
        f"{district_n}에는 전체 주유소 {int(r['stations'])}개 중 GS칼텍스가 {int(r['gs_stations'])}개로, GS Network 비중은 {r['gs_share']:.1f}%입니다.",
        f"이 비중은 서울 25개 구의 중앙값과 비교해 {share_desc} 보이며, 주유소당 등록차량은 {r['vehicles_per_station']:,.0f}대로 {vps_desc}",
        action_text,
        "다만 실제 확대·효율화 판단에는 주유소별 판매량, 인근 GS 점포 간 수요 중첩, 계약 확보 가능성, 예상 투자비와 물류조건을 추가로 확인해야 합니다.",
    ])

# ---------------------------------------------------------
# 4. Price
# ---------------------------------------------------------
with t4:
    st.subheader("가격·경쟁 포지셔닝")
    guide(
        "③ Network 수만으로는 경쟁력을 판단할 수 없습니다.",
        "같은 자치구에서도 소비자가 접하는 가격은 다르며, GS칼텍스가 지역·경쟁사 대비 어느 가격대에 위치하는지 확인해야 합니다.",
        "자치구를 선택해 전체 평균가격, GS칼텍스 평균가격, 브랜드별 평균가격과 중앙가격·최저·최고가격을 비교합니다.",
        "GS 가격이 자치구 평균보다 높거나 낮은지를 확인해 고객수용성·입지·서비스·프로모션·경쟁강도를 추가로 확인할 지역을 찾습니다. 가격 프리미엄 자체를 수익성으로 해석하지 않습니다.",
    )

    c1, c2 = st.columns(2)
    district_p = c1.selectbox("자치구", TARGET_DISTRICTS, key="price_district")
    fuel_p = c2.radio("제품", ["휘발유", "경유"], horizontal=True, key="price_fuel")
    _, price_col, _ = FUEL_COLS[fuel_p]
    pd_d = net[net["district"] == district_p].copy()
    valid = pd_d.dropna(subset=[price_col]).copy()
    brand_avg = valid.groupby("brand", as_index=False)[price_col].mean().sort_values(price_col)

    district_avg = valid[price_col].mean()
    gs_avg = valid.loc[valid["brand"] == "GS칼텍스", price_col].mean()
    gap = gs_avg - district_avg

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{district_p} 평균", "-" if pd.isna(district_avg) else f"{district_avg:,.0f}원/L")
    m2.metric("GS칼텍스 평균", "-" if pd.isna(gs_avg) else f"{gs_avg:,.0f}원/L")
    m3.metric("GS - 자치구 평균", "-" if pd.isna(gap) else f"{gap:+,.0f}원/L")

    fig = px.bar(brand_avg, x="brand", y=price_col, labels={"brand": "상표", price_col: "원/L"}, title=f"{district_p} 브랜드별 평균 {fuel_p} 가격")
    if pd.notna(district_avg):
        fig.add_hline(y=district_avg, line_dash="dash", annotation_text="자치구 평균")
    fig.update_layout(height=410, xaxis_title="", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 브랜드별 가격 수준을 어떻게 봐야 하나요?")
    st.caption("아래 표는 각 브랜드의 평균만이 아니라 중앙가격, 최저·최고가격과 관측 주유소 수를 함께 보여줍니다. 극단값 하나 때문에 분포가 왜곡되는 것을 피하기 위해 박스플롯 대신 숫자로 비교합니다.")
    brand_stats = (
        valid.groupby("brand")[price_col]
        .agg(["count", "min", "median", "mean", "max"])
        .reset_index()
        .sort_values("mean")
    )
    brand_stats.columns = ["브랜드", "주유소 수", "최저가격", "중앙가격", "평균가격", "최고가격"]
    st.dataframe(
        brand_stats.style.format({
            "주유소 수": "{:,.0f}", "최저가격": "{:,.0f}원/L", "중앙가격": "{:,.0f}원/L",
            "평균가격": "{:,.0f}원/L", "최고가격": "{:,.0f}원/L",
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"가격과 Network는 {SNAPSHOT_DATE} 기준 스냅샷입니다. 장기 가격추세가 아니라 분석시점의 경쟁 포지션을 보여줍니다.")

    if pd.isna(gs_avg):
        lines = [
            f"{district_p}에는 현재 저장 데이터 기준 GS칼텍스 {fuel_p} 가격을 계산할 수 있는 관측값이 없어 브랜드 평균 비교가 어렵습니다.",
            "따라서 이 지역은 가격 자체보다 GS Network 존재 여부와 개별 점포 데이터 확보 여부를 먼저 확인해야 합니다.",
            "경쟁 브랜드의 가격 수준은 표에서 확인할 수 있지만 GS와의 직접 격차를 계산해 전략을 제시하는 것은 적절하지 않습니다.",
            "실제 영업 검토에서는 최신 가격과 판매량, 주변 경쟁점 거리와 프로모션 조건을 추가로 확인해야 합니다.",
        ]
    else:
        direction = "낮은" if gap < 0 else "높은" if gap > 0 else "같은"
        meaning = (
            "지역 평균보다 낮은 가격대로 형성되어 있어 가격 경쟁 측면에서는 상대적으로 공격적인 위치로 볼 수 있습니다."
            if gap < 0 else
            "지역 평균보다 높은 가격대로 형성되어 있어 고객이 이 가격을 받아들이는 이유와 입지·서비스 차별성을 함께 점검할 필요가 있습니다."
            if gap > 0 else
            "지역 평균과 유사한 가격대로 형성되어 있어 가격 자체보다 입지와 서비스, 경쟁점 구성이 더 중요한 비교 요소가 될 수 있습니다."
        )
        lines = [
            f"{district_p}의 {fuel_p} 평균가격은 {district_avg:,.0f}원/L이고, GS칼텍스 평균은 {gs_avg:,.0f}원/L로 자치구 평균보다 {abs(gap):,.0f}원/L {direction} 수준입니다.",
            meaning,
            "브랜드별 중앙가격과 최저·최고가격을 함께 보면 평균값 하나만으로는 보이지 않는 경쟁 브랜드의 가격 폭과 포지션도 확인할 수 있습니다.",
            "다만 가격 차이만으로 판매성과나 수익성을 판단하지 않고, 실제 판매량·마진·입지·프로모션·주변 경쟁점 가격을 추가 확인해야 합니다.",
        ]
    conclusion_box(lines)

# ---------------------------------------------------------
# 5. Individual network
# ---------------------------------------------------------
with t5:
    st.subheader("개별 GS칼텍스 Network 진단")
    guide(
        "④ 자치구 분석에서 개별 Network로 내려갑니다.",
        "자치구 단위에서 검토대상을 찾은 뒤에는 같은 지역 안에서도 어떤 GS 주유소가 가격 측면에서 눈에 띄는지 확인해야 합니다.",
        "GS칼텍스 주유소를 선택해 해당 주유소 가격을 자치구 전체 평균, 자치구 GS 평균, 전체 가격분포와 비교합니다. 필요하면 오피넷 API로 최신 주변 경쟁정보를 별도로 조회할 수 있습니다.",
        "지역 평균보다 높은 가격을 바로 문제로 결론내리지 않고, 판매량·주변 경쟁점·입지·프로모션을 추가 확인할 개별 Network를 찾습니다.",
    )

    c1, c2 = st.columns(2)
    district_i = c1.selectbox("자치구", TARGET_DISTRICTS, key="station_district")
    fuel_i = c2.radio("제품", ["휘발유", "경유"], horizontal=True, key="station_fuel")
    _, price_i, prod_i = FUEL_COLS[fuel_i]

    gs_list = net[(net["district"] == district_i) & (net["brand"] == "GS칼텍스")].sort_values("station_name")
    if gs_list.empty:
        st.warning("선택 자치구에 GS칼텍스 주유소 데이터가 없습니다.")
        conclusion_box([
            f"{district_i}에는 현재 저장된 오피넷 스냅샷 기준 GS칼텍스 주유소가 없어 개별 Network 가격진단을 수행할 수 없습니다.",
            "이 경우 개별 점포의 가격 위치보다 해당 자치구에서 GS Network가 비어 있는 이유와 신규 Network 확보 가능성을 먼저 확인하는 것이 적절합니다.",
            "다만 Network 공백이 곧바로 출점 기회를 의미하는 것은 아니며 부지, 경쟁강도, 예상 판매량과 계약조건을 함께 검토해야 합니다.",
            "최종 판단은 Network 확대 검토 탭의 시장규모 지표와 내부 영업정보를 결합해 내려야 합니다.",
        ])
    else:
        station_name = st.selectbox("GS칼텍스 주유소", gs_list["station_name"].tolist())
        row = gs_list[gs_list["station_name"] == station_name].iloc[0]
        dd = net[net["district"] == district_i].dropna(subset=[price_i])
        sp = row[price_i]
        davg = dd[price_i].mean()
        gsavg = dd.loc[dd["brand"] == "GS칼텍스", price_i].mean()
        percentile = (dd[price_i].le(sp).mean() * 100) if pd.notna(sp) and not dd.empty else np.nan

        a, b, c, dcol = st.columns(4)
        a.metric("선택 주유소 가격", "-" if pd.isna(sp) else f"{sp:,.0f}원/L")
        b.metric("자치구 평균 대비", "-" if pd.isna(sp) else f"{sp-davg:+,.0f}원/L")
        c.metric("GS 평균 대비", "-" if pd.isna(sp) else f"{sp-gsavg:+,.0f}원/L")
        dcol.metric("자치구 내 가격 위치", price_position(percentile))
        st.write(f"주소: {row['address']}")

        compare = dd[["station_name", "brand", price_i]].sort_values(price_i).copy()
        st.dataframe(compare.rename(columns={"station_name": "주유소", "brand": "상표", price_i: f"{fuel_i} 가격"}), hide_index=True, use_container_width=True, height=320)

        if pd.isna(sp):
            lines = [
                f"{station_name}의 {fuel_i} 가격은 현재 저장 데이터에서 확인되지 않아 자치구 내 가격 위치를 계산할 수 없습니다.",
                "따라서 이 주유소는 가격 비교보다 최신 가격정보 확보가 우선입니다.",
                "주변 경쟁점과의 비교를 위해서는 오피넷 API 또는 동일 기준일의 추가 데이터를 확인할 수 있습니다.",
                "가격이 확보된 뒤 판매량·입지·프로모션과 함께 보아야 실제 영업 판단으로 연결할 수 있습니다.",
            ]
        else:
            higher_lower = "높습니다" if sp > davg else "낮습니다" if sp < davg else "같습니다"
            gs_higher_lower = "높습니다" if sp > gsavg else "낮습니다" if sp < gsavg else "같습니다"
            lines = [
                f"{station_name}의 {fuel_i} 가격은 {sp:,.0f}원/L로 {district_i} 평균보다 {abs(sp-davg):,.0f}원/L {higher_lower}",
                f"같은 자치구의 GS칼텍스 평균과 비교하면 {abs(sp-gsavg):,.0f}원/L {gs_higher_lower} 자치구 전체에서는 {price_position(percentile)}에 위치합니다.",
                "따라서 선택 주유소가 지역 평균보다 높은 가격대를 유지한다면 입지·서비스·고객충성도·주변 경쟁점 가격이 이를 설명하는지 추가로 확인할 필요가 있습니다.",
                "반대로 낮은 가격대라면 판매량 확대 효과와 마진 부담을 함께 확인해야 하며, 가격 수준만으로 성과를 단정하지 않습니다.",
            ]
        conclusion_box(lines)

    st.markdown("#### 최신 주변 경쟁정보 · 오피넷 API (선택 기능)")
    st.caption("핵심 분석은 저장된 2026.09.12 데이터만으로 작동합니다. API는 최신 경쟁정보를 확인할 때만 쓰는 보조기능입니다.")
    api_key = get_api_key()
    if not api_key:
        st.info("API Key가 없어도 기본 분석은 정상 작동합니다. 필요하면 Streamlit Secrets에 OPINET_API_KEY를 설정하세요.")
    else:
        with st.expander("최신 주변 경쟁주유소 조회", expanded=False):
            colx, coly, colr = st.columns(3)
            x = colx.number_input("KATEC X", value=0.0, format="%.3f")
            y = coly.number_input("KATEC Y", value=0.0, format="%.3f")
            radius = colr.select_slider("반경(m)", options=[1000, 2000, 3000, 4000, 5000], value=3000)
            if st.button("최신 경쟁정보 조회", type="secondary"):
                if x == 0 or y == 0:
                    st.warning("KATEC X/Y 좌표를 입력하세요.")
                else:
                    try:
                        live = opinet_around(api_key, x, y, radius, prod_i)
                        if live.empty:
                            st.info("해당 조건의 조회 결과가 없습니다.")
                        else:
                            cols = [c for c in ["OS_NM", "브랜드", "PRICE", "DISTANCE", "UNI_ID"] if c in live.columns]
                            st.dataframe(live[cols].rename(columns={"OS_NM": "주유소", "PRICE": "가격", "DISTANCE": "거리(m)", "UNI_ID": "주유소ID"}), hide_index=True, use_container_width=True)
                    except Exception:
                        st.warning("최신 경쟁정보 조회를 완료하지 못했습니다. 저장된 데이터 기반 분석에는 영향이 없습니다.")

# ---------------------------------------------------------
# 6. Final conclusion
# ---------------------------------------------------------
with t6:
    st.subheader("최종결론 · 서울 25개 구 영업 검토 우선순위")
    guide(
        "⑤ 마지막에는 데이터를 영업의 다음 행동으로 바꿉니다.",
        "대시보드가 숫자 나열에 그치지 않도록 앞선 수요·시장규모·Network·가격 지표를 종합해 25개 구 중 어디부터 더 깊게 볼지 정합니다.",
        "Network 확대는 낮은 GS 비중·높은 주유소당 등록차량·차량 증가세를, 기존 Network 효율화는 높은 GS 비중·낮은 주유소당 등록차량을, 가격점검은 GS의 자치구 평균 대비 가격 프리미엄을 상대순위로 비교합니다.",
        "각 영역의 1순위와 상위 후보를 제시하되, 이는 출점·철수·가격변경의 확정 결론이 아니라 내부 수익성 데이터를 추가 확인할 우선순위입니다.",
    )

    st.markdown("#### 스크리닝 점수는 무엇인가요?")
    st.info(
        "확대 검토점수는 GS Network 비중이 낮을수록, 주유소당 등록차량이 많을수록, 자동차 등록대수 증가율이 높을수록 점수가 올라갑니다. "
        "효율화 검토점수는 GS Network 비중이 높을수록, 주유소당 등록차량이 적을수록 점수가 올라갑니다. "
        "가격점검 순위는 GS칼텍스 평균가격이 자치구 평균보다 얼마나 높은지를 상대 비교한 것입니다. 모든 점수는 실제 수익성 점수가 아니라 추가 확인 순서를 정하기 위한 상대적 스크리닝 지표입니다."
    )

    expand_top = summary.sort_values("expansion_score", ascending=False).head(5)
    eff_top = summary.sort_values("efficiency_score", ascending=False).head(5)
    price_top = summary.dropna(subset=["avg_price_gap"]).sort_values("price_watch_score", ascending=False).head(5)

    a, b, c = st.columns(3)
    with a:
        top_exp = expand_top.iloc[0]
        st.metric("Network 확대 검토 1순위", top_exp["district"], f"스크리닝 {top_exp['expansion_score']:.0f}점")
        st.caption(f"GS 비중 {top_exp['gs_share']:.1f}% · 주유소당 차량 {top_exp['vehicles_per_station']:,.0f}대 · 자동차 YoY {top_exp['vehicle_yoy']:+.2f}%")
    with b:
        top_eff = eff_top.iloc[0]
        st.metric("기존 Network 효율화 1순위", top_eff["district"], f"스크리닝 {top_eff['efficiency_score']:.0f}점")
        st.caption(f"GS 비중 {top_eff['gs_share']:.1f}% · 주유소당 차량 {top_eff['vehicles_per_station']:,.0f}대")
    with c:
        if not price_top.empty:
            top_price = price_top.iloc[0]
            st.metric("가격 경쟁 점검 1순위", top_price["district"], f"평균 프리미엄 {top_price['avg_price_gap']:+.0f}원/L")
            st.caption(f"휘발유 {top_price['gs_gasoline_gap']:+.0f}원/L · 경유 {top_price['gs_diesel_gap']:+.0f}원/L")

    st.markdown("### 1. Network 확대 검토 후보")
    exp_show = expand_top[["district", "gs_share", "vehicles_per_station", "vehicle_yoy", "expansion_score"]].copy()
    exp_show.columns = ["자치구", "GS Network 비중(%)", "주유소당 등록차량", "자동차 YoY(%)", "확대 검토점수"]
    st.dataframe(exp_show.style.format({"GS Network 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}", "자동차 YoY(%)": "{:+.2f}", "확대 검토점수": "{:.0f}"}), hide_index=True, use_container_width=True)
    e1 = expand_top.iloc[0]
    e2 = expand_top.iloc[1] if len(expand_top) > 1 else e1
    st.markdown(
        f"""
        <div class="conclusion-box">
        확대 후보는 GS Network 비중이 낮으면서 주유소당 등록차량이 많고 차량시장 증가세가 상대적으로 높은 지역을 우선 선별한 결과입니다.<br>
        현재 1순위는 {e1['district']}로, GS Network 비중 {e1['gs_share']:.1f}%, 주유소당 등록차량 {e1['vehicles_per_station']:,.0f}대, 자동차 YoY {e1['vehicle_yoy']:+.2f}%입니다. 단순히 GS 비중이 낮다는 이유만이 아니라 시장규모와 성장 방향을 함께 본 결과입니다.<br>
        상위 후보 중 GS 주유소가 0개인 지역은 Network 공백 자체가 추가 검토 사유가 되지만, 기존 Network가 없다는 사실만으로 신규 출점을 정당화하지는 않습니다. 예상 판매량, 경쟁점 거리, 부지·계약 확보 가능성을 함께 확인해야 합니다.<br>
        따라서 확대 후보 표는 신규 출점 확정표가 아니라 영업 담당자가 어디부터 상권·교통량·판매잠재력·브랜드 전환 가능성을 조사할지 정하는 우선순위로 해석합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 2. 기존 Network 효율화 검토 후보")
    eff_show = eff_top[["district", "gs_share", "vehicles_per_station", "efficiency_score"]].copy()
    eff_show.columns = ["자치구", "GS Network 비중(%)", "주유소당 등록차량", "효율화 검토점수"]
    st.dataframe(eff_show.style.format({"GS Network 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}", "효율화 검토점수": "{:.0f}"}), hide_index=True, use_container_width=True)
    f1 = eff_top.iloc[0]
    st.markdown(
        f"""
        <div class="conclusion-box">
        효율화 후보는 GS칼텍스 Network 비중이 이미 상대적으로 높으면서 주유소당 등록차량이 낮은 지역을 우선 선별한 결과입니다.<br>
        현재 1순위는 {f1['district']}로, GS Network 비중은 {f1['gs_share']:.1f}%이고 주유소당 등록차량은 {f1['vehicles_per_station']:,.0f}대입니다. 공개자료만 보면 신규 Network 추가보다 현재 점포별 판매성과와 수요 중첩 여부를 먼저 점검할 필요성이 상대적으로 높습니다.<br>
        다만 주유소당 등록차량이 낮다는 사실만으로 비효율이라고 단정할 수 없습니다. 유동교통량, 법인·사업용 수요, 점포별 판매량, 마진과 물류조건에 따라 실제 성과는 달라질 수 있습니다.<br>
        따라서 효율화 후보 표는 철수 후보가 아니라 기존 Network 안에서 판매량·비용·점포 간 수요잠식 여부를 우선 확인할 지역의 순서입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 3. 가격 경쟁 추가점검 후보")
    if not price_top.empty:
        price_show = price_top[["district", "gs_gasoline_gap", "gs_diesel_gap", "avg_price_gap"]].copy()
        price_show.columns = ["자치구", "GS 휘발유 가격차", "GS 경유 가격차", "평균 가격차"]
        st.dataframe(price_show.style.format({"GS 휘발유 가격차": "{:+.0f}원/L", "GS 경유 가격차": "{:+.0f}원/L", "평균 가격차": "{:+.0f}원/L"}), hide_index=True, use_container_width=True)
        p1 = price_top.iloc[0]
        st.markdown(
            f"""
            <div class="conclusion-box">
            가격 점검 후보는 GS칼텍스 평균가격이 해당 자치구 전체 평균보다 상대적으로 높은 지역을 우선 선별한 결과입니다.<br>
            현재 1순위는 {p1['district']}로, GS칼텍스는 자치구 평균 대비 휘발유 {p1['gs_gasoline_gap']:+.0f}원/L, 경유 {p1['gs_diesel_gap']:+.0f}원/L의 차이를 보입니다. 이는 가격경쟁 위치를 추가로 확인할 필요가 있다는 의미입니다.<br>
            가격 프리미엄이 크다고 바로 가격을 내려야 한다고 결론내리지는 않습니다. 입지, 서비스, 고객충성도, 판촉, 주변 경쟁점 가격이 높은 가격을 정당화할 수도 있기 때문입니다.<br>
            실제 영업에서는 가격변경 전 판매량 반응, 마진, 경쟁점 가격, 프로모션 효과를 함께 확인해 가격전략의 적정성을 판단해야 합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("가격 비교가 가능한 GS칼텍스 데이터가 충분하지 않습니다.")

    st.markdown("### 4. 자치구별 종합판단")
    final_table = summary[["district", "primary_action", "gs_share", "vehicles_per_station", "vehicle_yoy", "gs_gasoline_gap", "gs_diesel_gap"]].copy()
    final_table.columns = ["자치구", "1차 판단", "GS Network 비중(%)", "주유소당 등록차량", "자동차 YoY(%)", "휘발유 가격차", "경유 가격차"]
    st.dataframe(final_table.style.format({"GS Network 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}", "자동차 YoY(%)": "{:+.2f}", "휘발유 가격차": "{:+.0f}원/L", "경유 가격차": "{:+.0f}원/L"}), hide_index=True, use_container_width=True, height=500)

    selected = st.selectbox("자치구별 결론 자세히 보기", TARGET_DISTRICTS, key="conclusion_district")
    r = summary[summary["district"] == selected].iloc[0]
    st.markdown(f"#### {selected} · {r['primary_action']}")
    st.write(
        f"{selected}에는 전체 주유소가 {int(r['stations'])}개 있고, 이 중 GS칼텍스는 {int(r['gs_stations'])}개({r['gs_share']:.1f}%)입니다. "
        f"최근 자동차 등록대수는 {r['vehicles']:,.0f}대, 주유소 1곳당 등록차량은 {r['vehicles_per_station']:,.0f}대이며 자동차 등록대수는 전년 동월 대비 {r['vehicle_yoy']:+.2f}%입니다. "
        f"GS칼텍스 평균가격은 자치구 평균 대비 휘발유 {r['gs_gasoline_gap']:+.0f}원/L, 경유 {r['gs_diesel_gap']:+.0f}원/L 수준입니다."
    )

    if r["primary_action"] == "Network 확대 우선검토":
        action_detail = "시장규모 대비 GS Network 침투가 상대적으로 낮아 신규 출점, 브랜드 전환, 계약 확보 가능성을 먼저 검토할 지역으로 분류됩니다."
    elif r["primary_action"] == "기존 Network 효율화 우선검토":
        action_detail = "GS Network가 상대적으로 충분한 편이므로 신규 확대보다 현재 점포의 판매량, 비용, 수요 중첩과 운영성과를 먼저 확인할 지역으로 분류됩니다."
    else:
        action_detail = "Network 확대나 효율화 한 방향으로 단정하기보다 가격, 입지, 경쟁점과 개별 주유소 성과를 더 확인해야 할 지역으로 분류됩니다."

    st.markdown(
        f"""
        <div class="conclusion-box">
        {action_detail}<br>
        이 판단은 GS Network 비중과 주유소당 등록차량을 중심으로 만든 1차 분류이므로 실제 수익성이나 점포성과를 직접 의미하지 않습니다.<br>
        가격차가 함께 크게 나타나는 경우에는 Network 규모 판단과 별도로 고객수용성·경쟁점 가격·프로모션을 추가 점검할 필요가 있습니다.<br>
        최종 의사결정 전에는 주유소별 판매량, 공급가격과 마진, 물류비, 판촉·시설지원비, 여신·채권, 계약조건, 신규점 투자비와 기존 Network 판매잠식 여부를 반드시 연결해야 합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 프로그램의 최종 메시지")
    st.success(
        "서울 전체 수요에서 시작해 25개 구의 시장규모, GS Network 침투도, 가격 경쟁을 연결하고 개별 Network까지 내려가 어디를 먼저 더 깊게 볼 것인가를 구조화했습니다. 공개자료로 수익성을 임의 추정하지 않고 실제 국내영업 의사결정의 검토 순서를 구현한 것이 핵심입니다."
    )
