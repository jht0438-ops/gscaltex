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
      .guide-box {padding: 1rem 1.1rem; border-radius: .7rem; background: rgba(120,120,120,.08); margin-bottom: 1rem;}
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
    net["self_service"] = net["self_service"].astype(str).str.upper().eq("Y")
    return cons, veh, net


cons, veh, net = load_data()

# =========================================================
# Helpers
# =========================================================
def district_vehicle_series(district: str) -> pd.DataFrame:
    d = veh[(veh["sido"] == "서울") & (veh["sigungu"] == district)].copy()
    if d.empty:
        return pd.DataFrame(columns=["month", "count"])

    # 전처리 파일에 '총계'가 있으면 이를 사용. 없을 때만 차종을 합산한다.
    if (d["vehicle_type"] == "총계").any():
        d = d[d["vehicle_type"] == "총계"]
        d = d.groupby("month", as_index=False)["count"].sum()
    else:
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
            "self_share": d["self_service"].mean() * 100 if total else np.nan,
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

    # 상대 비교용 점수: 실제 수익성 점수가 아니라 '추가 검토 순서'를 만들기 위한 스크리닝 점수
    share_low = safe_rank_pct(out["gs_share"], ascending=False)  # 낮을수록 점수 높음
    vps_high = safe_rank_pct(out["vehicles_per_station"], ascending=True)
    yoy_high = safe_rank_pct(out["vehicle_yoy"].fillna(out["vehicle_yoy"].median()), ascending=True)
    out["expansion_score"] = (share_low + vps_high + yoy_high) / 3

    share_high = safe_rank_pct(out["gs_share"], ascending=True)
    vps_low = safe_rank_pct(out["vehicles_per_station"], ascending=False)
    out["efficiency_score"] = (share_high + vps_low) / 2

    out["price_watch_score"] = safe_rank_pct(out["avg_price_gap"].fillna(out["avg_price_gap"].median()), ascending=True)

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

# =========================================================
# Header
# =========================================================
st.title("⛽ GS칼텍스 서울 주유소 Network 분석 및 의사결정 지원 도구")
st.caption(f"서울특별시 25개 자치구 · 오피넷 Network/가격 기준일 {SNAPSHOT_DATE} · 공개자료 기반")

with st.expander("분석 원칙과 공개자료의 한계", expanded=False):
    st.markdown(
        """
- **목적:** 서울 전체 석유제품 수요와 25개 자치구의 차량시장·주유소 Network·가격을 연결해 영업 담당자가 추가 검토할 지역을 찾습니다.
- **실제 수익성 계산 아님:** 주유소별 판매량, GS칼텍스 공급가격, 물류비, 판촉·지원비, 여신·채권, 계약조건이 없으므로 실제 이익을 추정하지 않습니다.
- **분석 단위:** 서울 전체 수요 → 자치구 시장 → 자치구 Network → 가격 경쟁 → 개별 GS Network로 범위를 좁힙니다.
- **가격/Network 데이터:** 오피넷에서 내려받은 **2026년 9월 12일 기준 스냅샷**입니다. 가격의 장기 추세가 아니라 분석시점의 경쟁 포지션을 봅니다.
- **최종 점수:** 절대적인 출점/철수 결론이 아니라 25개 구를 상대 비교해 추가 검토 순서를 만드는 스크리닝 지표입니다.
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
    st.write(
        "이 프로그램의 핵심 질문은 **'서울 25개 자치구 중 GS칼텍스가 Network 확대를 먼저 검토할 곳과, "
        "기존 Network의 효율성을 먼저 점검할 곳은 어디인가?'**입니다. 공개자료만으로 실제 수익성을 단정하지 않고, "
        "시장 수요에서 시작해 Network와 가격, 개별 주유소까지 단계적으로 범위를 좁혀 영업 검토 우선순위를 만듭니다."
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
            "브랜드별 주유소 수 + 셀프 여부",
            "브랜드별 휘발유·경유 가격",
            "개별 주유소 가격·주소·운영형태",
            "앞선 지표의 상대비교 결과",
        ],
    })
    st.dataframe(flow, hide_index=True, use_container_width=True)

    st.markdown("### 이 분석이 도달하려는 결론")
    st.info(
        "시장규모에 비해 GS칼텍스 Network 비중이 낮은 자치구는 **Network 확대 후보**, 이미 GS Network가 상대적으로 충분하지만 "
        "주유소당 잠재 차량시장이 작은 곳은 **기존 Network 효율화 후보**, GS 가격 프리미엄이 큰 곳은 **가격 경쟁 추가점검 후보**로 구분합니다. "
        "이후 실제 의사결정에서는 내부 판매량·공급가격·물류비·판촉비·여신/채권·계약조건을 연결해 수익성을 검증해야 합니다."
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
        "서울 전체의 수요 흐름과 함께, 어떤 구가 상대적으로 큰 차량시장과 높은 Network 수용여력을 갖는지 확인합니다.",
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

# ---------------------------------------------------------
# 3. Network
# ---------------------------------------------------------
with t3:
    st.subheader("서울 25개 자치구 Network 구조")
    guide(
        "② 시장규모 다음에는 Network 침투도를 봅니다.",
        "시장규모가 커도 GS칼텍스 Network가 이미 충분히 확보된 곳과 상대적으로 부족한 곳은 영업전략이 달라야 합니다.",
        "자치구별 전체 주유소와 브랜드별 주유소 수, GS칼텍스 Network 비중, 셀프 운영비중을 비교하고 앞 탭의 주유소당 등록차량과 함께 봅니다.",
        "시장규모 대비 GS Network가 상대적으로 부족한 구와, 이미 높은 비중을 확보한 구를 구분해 확대와 효율화의 후보군을 만듭니다.",
    )

    cnt = net.groupby(["district", "brand"]).size().reset_index(name="주유소수")
    fig = px.bar(cnt, x="district", y="주유소수", color="brand", barmode="stack", title="서울 자치구별 브랜드 Network 구성")
    fig.update_layout(height=470, xaxis_title="", xaxis_tickangle=-45, legend_title="상표", margin=dict(l=10, r=10, t=55, b=85))
    st.plotly_chart(fig, use_container_width=True)

    show = summary[["district", "stations", "gs_stations", "gs_share", "self_share", "vehicles_per_station"]].sort_values("gs_share", ascending=False).copy()
    show.columns = ["자치구", "전체 주유소", "GS칼텍스", "GS Network 비중(%)", "셀프 비중(%)", "주유소당 등록차량"]
    st.dataframe(
        show.style.format({"전체 주유소": "{:,.0f}", "GS칼텍스": "{:,.0f}", "GS Network 비중(%)": "{:.1f}", "셀프 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}"}),
        hide_index=True, use_container_width=True, height=490,
    )

    district_n = st.selectbox("운영형태를 자세히 볼 자치구", TARGET_DISTRICTS, key="network_district")
    nd = net[net["district"] == district_n]
    brand_self = nd.groupby("brand", as_index=False)["self_service"].mean()
    brand_self["셀프비중"] = brand_self["self_service"] * 100
    fig_self = px.bar(brand_self.sort_values("셀프비중"), x="셀프비중", y="brand", orientation="h", labels={"brand": "상표", "셀프비중": "셀프 비중(%)"}, title=f"{district_n} 브랜드별 셀프주유소 비중")
    fig_self.update_layout(height=350, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig_self, use_container_width=True)
    st.caption("셀프 비중은 운영형태를 설명하는 보조지표입니다. 인건비·투자비·판매량이 없으므로 셀프 비중만으로 운영효율을 단정하지 않습니다.")

# ---------------------------------------------------------
# 4. Price
# ---------------------------------------------------------
with t4:
    st.subheader("가격·경쟁 포지셔닝")
    guide(
        "③ Network 수만으로는 경쟁력을 판단할 수 없습니다.",
        "같은 자치구에서도 소비자가 접하는 가격은 다르며, GS칼텍스가 지역·경쟁사 대비 어느 가격대에 위치하는지 확인해야 합니다.",
        "자치구를 선택해 전체 평균가격, GS칼텍스 평균가격, 브랜드별 평균과 가격분포를 비교합니다.",
        "GS 가격 프리미엄이 큰 곳을 찾아 고객수용성·입지·서비스·프로모션·경쟁강도를 추가로 확인할 후보로 분류합니다. 가격 프리미엄 자체를 수익성으로 해석하지 않습니다.",
    )

    c1, c2 = st.columns(2)
    district_p = c1.selectbox("자치구", TARGET_DISTRICTS, key="price_district")
    fuel_p = c2.radio("제품", ["휘발유", "경유"], horizontal=True, key="price_fuel")
    _, price_col, _ = FUEL_COLS[fuel_p]
    pd_d = net[net["district"] == district_p].copy()
    brand_avg = pd_d.groupby("brand", as_index=False)[price_col].mean().dropna().sort_values(price_col)

    district_avg = pd_d[price_col].mean()
    gs_avg = pd_d.loc[pd_d["brand"] == "GS칼텍스", price_col].mean()
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

    fig_box = px.box(pd_d.dropna(subset=[price_col]), x="brand", y=price_col, points="outliers", labels={"brand": "상표", price_col: "원/L"}, title="브랜드별 가격 분포")
    fig_box.update_layout(height=420, xaxis_title="", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(f"가격과 Network는 {SNAPSHOT_DATE} 기준 스냅샷입니다. 장기 가격추세가 아니라 분석시점의 경쟁 포지션을 보여줍니다.")

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
        dcol.metric("자치구 가격 백분위", "-" if pd.isna(percentile) else f"{percentile:.0f}%")
        st.write(f"**주소:** {row['address']}  ·  **셀프:** {'Y' if row['self_service'] else 'N'}")

        compare = dd[["station_name", "brand", price_i, "self_service"]].sort_values(price_i).copy()
        st.dataframe(compare.rename(columns={"station_name": "주유소", "brand": "상표", price_i: f"{fuel_i} 가격", "self_service": "셀프"}), hide_index=True, use_container_width=True, height=320)

    st.markdown("#### 최신 주변 경쟁정보 · 오피넷 API (선택 기능)")
    st.caption("핵심 분석은 저장된 2026.09.12 데이터만으로 작동합니다. API는 최신 경쟁정보를 확인할 때만 쓰는 보조기능입니다.")
    api_key = get_api_key()
    if not api_key:
        st.info("API Key가 없어도 기본 분석은 정상 작동합니다. 필요하면 Streamlit Secrets에 `OPINET_API_KEY`를 설정하세요.")
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
        "⑤ 마지막에는 데이터를 '영업의 다음 행동'으로 바꿉니다.",
        "대시보드가 숫자 나열에 그치지 않도록 앞선 수요·시장규모·Network·가격 지표를 종합해 25개 구 중 어디부터 더 깊게 볼지 정합니다.",
        "Network 확대는 낮은 GS 비중·높은 주유소당 등록차량·차량 증가세를, 기존 Network 효율화는 높은 GS 비중·낮은 주유소당 등록차량을, 가격점검은 GS의 자치구 평균 대비 가격 프리미엄을 상대순위로 비교합니다.",
        "각 영역의 1순위와 상위 후보를 제시하되, 이는 출점·철수·가격변경의 확정 결론이 아니라 내부 수익성 데이터를 추가 확인할 우선순위입니다.",
    )

    expand_top = summary.sort_values("expansion_score", ascending=False).head(5)
    eff_top = summary.sort_values("efficiency_score", ascending=False).head(5)
    price_top = summary.dropna(subset=["avg_price_gap"]).sort_values("price_watch_score", ascending=False).head(5)

    a, b, c = st.columns(3)
    with a:
        top = expand_top.iloc[0]
        st.metric("Network 확대 검토 1순위", top["district"], f"스크리닝 {top['expansion_score']:.0f}점")
        st.caption(f"GS 비중 {top['gs_share']:.1f}% · 주유소당 차량 {top['vehicles_per_station']:,.0f}대 · 자동차 YoY {top['vehicle_yoy']:+.2f}%")
    with b:
        top = eff_top.iloc[0]
        st.metric("기존 Network 효율화 1순위", top["district"], f"스크리닝 {top['efficiency_score']:.0f}점")
        st.caption(f"GS 비중 {top['gs_share']:.1f}% · 주유소당 차량 {top['vehicles_per_station']:,.0f}대")
    with c:
        if not price_top.empty:
            top = price_top.iloc[0]
            st.metric("가격 경쟁 점검 1순위", top["district"], f"평균 프리미엄 {top['avg_price_gap']:+.0f}원/L")
            st.caption(f"휘발유 {top['gs_gasoline_gap']:+.0f}원/L · 경유 {top['gs_diesel_gap']:+.0f}원/L")

    st.markdown("### 1. Network 확대 검토 후보")
    exp_show = expand_top[["district", "gs_share", "vehicles_per_station", "vehicle_yoy", "expansion_score"]].copy()
    exp_show.columns = ["자치구", "GS Network 비중(%)", "주유소당 등록차량", "자동차 YoY(%)", "확대 검토점수"]
    st.dataframe(exp_show.style.format({"GS Network 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}", "자동차 YoY(%)": "{:+.2f}", "확대 검토점수": "{:.0f}"}), hide_index=True, use_container_width=True)
    st.caption("GS 비중이 상대적으로 낮고, 주유소당 차량시장과 차량 증가세가 높은 구일수록 상단에 위치합니다.")

    st.markdown("### 2. 기존 Network 효율화 검토 후보")
    eff_show = eff_top[["district", "gs_share", "vehicles_per_station", "efficiency_score"]].copy()
    eff_show.columns = ["자치구", "GS Network 비중(%)", "주유소당 등록차량", "효율화 검토점수"]
    st.dataframe(eff_show.style.format({"GS Network 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}", "효율화 검토점수": "{:.0f}"}), hide_index=True, use_container_width=True)
    st.caption("GS 비중이 상대적으로 높으면서 주유소당 잠재 차량시장이 낮은 구는 신규 확대보다 기존 Network의 판매량·비용·운영성과를 먼저 점검하는 방향으로 해석합니다.")

    st.markdown("### 3. 가격 경쟁 추가점검 후보")
    if not price_top.empty:
        price_show = price_top[["district", "gs_gasoline_gap", "gs_diesel_gap", "avg_price_gap"]].copy()
        price_show.columns = ["자치구", "GS 휘발유 가격차", "GS 경유 가격차", "평균 가격차"]
        st.dataframe(price_show.style.format({"GS 휘발유 가격차": "{:+.0f}원/L", "GS 경유 가격차": "{:+.0f}원/L", "평균 가격차": "{:+.0f}원/L"}), hide_index=True, use_container_width=True)
    st.caption("가격 프리미엄이 크다는 사실만으로 가격인하를 제안하지 않습니다. 고객수용성·입지·서비스·프로모션·경쟁점 가격을 추가 확인해야 한다는 의미입니다.")

    st.markdown("### 4. 자치구별 종합판단")
    final_table = summary[["district", "primary_action", "gs_share", "vehicles_per_station", "vehicle_yoy", "gs_gasoline_gap", "gs_diesel_gap"]].copy()
    final_table.columns = ["자치구", "1차 판단", "GS Network 비중(%)", "주유소당 등록차량", "자동차 YoY(%)", "휘발유 가격차", "경유 가격차"]
    st.dataframe(final_table.style.format({"GS Network 비중(%)": "{:.1f}", "주유소당 등록차량": "{:,.0f}", "자동차 YoY(%)": "{:+.2f}", "휘발유 가격차": "{:+.0f}원/L", "경유 가격차": "{:+.0f}원/L"}), hide_index=True, use_container_width=True, height=500)

    selected = st.selectbox("자치구별 결론 자세히 보기", TARGET_DISTRICTS, key="conclusion_district")
    r = summary[summary["district"] == selected].iloc[0]
    st.markdown(f"#### {selected} · {r['primary_action']}")
    st.write(
        f"{selected}에는 전체 주유소가 **{int(r['stations'])}개** 있고, 이 중 GS칼텍스는 **{int(r['gs_stations'])}개({r['gs_share']:.1f}%)**입니다. "
        f"최근 자동차 등록대수는 **{r['vehicles']:,.0f}대**, 주유소 1곳당 등록차량은 **{r['vehicles_per_station']:,.0f}대**이며 자동차 등록대수는 전년 동월 대비 **{r['vehicle_yoy']:+.2f}%**입니다. "
        f"GS칼텍스 평균가격은 자치구 평균 대비 휘발유 **{r['gs_gasoline_gap']:+.0f}원/L**, 경유 **{r['gs_diesel_gap']:+.0f}원/L** 수준입니다."
    )

    if r["primary_action"] == "Network 확대 우선검토":
        st.success("공개자료 단계에서는 시장규모 대비 GS Network 침투가 상대적으로 낮은지를 우선 확인할 후보입니다. 신규 출점뿐 아니라 기존 주유소의 브랜드 전환·계약 확보 가능성도 검토대상입니다.")
    elif r["primary_action"] == "기존 Network 효율화 우선검토":
        st.info("공개자료 단계에서는 신규 Network 추가보다 현재 보유 Network의 판매량·비용·운영성과를 먼저 점검할 후보입니다.")
    else:
        st.info("Network 확대/효율화 한 방향으로 단정하기보다 개별 주유소의 가격·입지·경쟁환경을 추가 확인하는 것이 우선인 구입니다.")

    st.warning(
        "**최종 의사결정 전 추가 검증:** 주유소별 실제 판매량, GS칼텍스 공급가격과 마진, 물류비, 판촉·시설지원비, 여신·채권, 계약조건, 신규점 투자비와 기존 Network 판매잠식 여부를 연결해야 합니다."
    )

    st.markdown("### 프로그램의 최종 메시지")
    st.success(
        "서울 전체 수요에서 시작해 25개 구의 시장규모·GS Network 침투도·가격 경쟁을 연결하고, 개별 Network까지 내려가 '어디를 먼저 더 깊게 볼 것인가'를 구조화했습니다. 공개자료로 수익성을 임의 추정하지 않고 실제 국내영업 의사결정의 검토 순서를 구현한 것이 핵심입니다."
    )
