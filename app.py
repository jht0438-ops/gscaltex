from pathlib import Path
import io
import re
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================================================
# Page / constants
# =========================================================
st.set_page_config(
    page_title="GS칼텍스 국내영업 Network·유외수익 분석",
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

st.markdown("""
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.18); padding:12px 14px; border-radius:12px;}
.guide-box {padding:1rem 1.1rem; border-radius:.7rem; background:rgba(120,120,120,.08); margin-bottom:1rem; line-height:1.75;}
.conclusion-box {padding:1rem 1.1rem; border-left:4px solid rgba(80,120,180,.85); background:rgba(120,120,120,.055); border-radius:.35rem; margin-top:1rem; line-height:1.8;}
.small-note {font-size:.88rem; opacity:.78;}
.section-text {line-height:1.85; font-size:1rem;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# File helpers
# =========================================================
def find_file(filename: str):
    """
    사용자가 제공한 원자료를 우선 탐색한다.
    - 배포용: ./data, 앱과 같은 폴더
    - ChatGPT 작업환경: /mnt/data
    - 파일명이 완전히 같지 않아도 공백/중복표시가 있는 경우 glob으로 보완
    """
    roots = [DATA_DIR, BASE_DIR, Path("/mnt/data")]
    for root in roots:
        p = root / filename
        if p.exists():
            return p

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for root in roots:
        if not root.exists():
            continue
        candidates = list(root.glob(f"*{stem.split(' (')[0]}*{suffix}"))
        if candidates:
            candidates = sorted(candidates, key=lambda x: (len(x.name), x.name))
            return candidates[0]
    return None

def read_csv_kr(path, **kwargs):
    last = None
    for enc in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last = e
    raise last

def read_html_first(path):
    tables = pd.read_html(path)
    if not tables:
        return pd.DataFrame()
    return tables[0]

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def latest_quarter(df, code_col="기준_년분기_코드"):
    if df.empty or code_col not in df.columns:
        return df.copy()
    code = safe_num(df[code_col])
    if code.notna().any():
        return df[code == code.max()].copy()
    return df.copy()

def guide(title, why, how, conclusion):
    st.markdown(
        f"""<div class="guide-box"><b>{title}</b><br><br>
        <b>왜 보나요?</b> {why}<br><br>
        <b>어떻게 보나요?</b> {how}<br><br>
        <b>무엇을 판단하나요?</b> {conclusion}</div>""",
        unsafe_allow_html=True,
    )

def conclusion_box(lines):
    st.markdown("### 이 탭에서 얻을 수 있는 결론은 무엇인가요?")
    st.markdown('<div class="conclusion-box">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

def fmt_num(x, digits=0):
    if pd.isna(x):
        return "확인 불가"
    return f"{x:,.{digits}f}" if digits else f"{x:,.0f}"

def pct_rank_high(s):
    return s.rank(pct=True, ascending=True, method="average") * 100

def pct_rank_low(s):
    return s.rank(pct=True, ascending=False, method="average") * 100

# =========================================================
# Core normalized data used by the previous version
# =========================================================
@st.cache_data
def load_core():
    files = {
        "cons": find_file("consumption_monthly.csv"),
        "veh": find_file("vehicles_monthly.csv"),
        "net": find_file("network_snapshot.csv"),
    }
    if not all(files.values()):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cons = pd.read_csv(files["cons"])
    veh = pd.read_csv(files["veh"])
    net = pd.read_csv(files["net"])
    cons["month"] = pd.to_datetime(cons["month"].astype(str) + "-01", errors="coerce")
    veh["month"] = pd.to_datetime(veh["month"].astype(str) + "-01", errors="coerce")
    for c in ["gasoline_price", "diesel_price", "premium_gasoline_price", "kerosene_price"]:
        if c in net.columns:
            net[c] = safe_num(net[c])
    return cons, veh, net

cons, veh, net = load_core()
CORE_OK = not cons.empty and not veh.empty and not net.empty

# =========================================================
# New raw datasets
# =========================================================
def parse_week_label(x):
    m = re.search(r"(?:(20)?(\d{2})년)?(\d{1,2})월(\d)주", str(x))
    if not m:
        return pd.NaT
    yy = int(m.group(2))
    year = 2000 + yy
    month = int(m.group(3))
    week = int(m.group(4))
    # 주간 비교용 대표일. 정확한 일별 가격을 뜻하지 않음.
    return pd.Timestamp(year, month, 1) + pd.Timedelta(days=(week - 1) * 7)

@st.cache_data
def load_price_cap_data():
    intl_path = find_file("국제_석유제품가격2026011-2026092.xlsx") or find_file("국제_석유제품가격2026011-2026092.xls")
    gas_path = find_file("정유사_주간공급가격_회사별.xls")
    diesel_path = find_file("정유사_주간공급가격_회사별 (1).xls")
    retail_path = find_file("주유소_제품별_평균판매가격.xls")

    intl = pd.DataFrame()
    if intl_path:
        try:
            raw = pd.read_excel(intl_path, header=None)
            header_idx = raw.index[raw.iloc[:, 0].astype(str).eq("기간")]
            if len(header_idx):
                h = int(header_idx[0])
                intl = raw.iloc[h + 1:].copy()
                intl.columns = raw.iloc[h].tolist()
                intl = intl.dropna(how="all")
            else:
                intl = pd.read_excel(intl_path)
        except Exception:
            try:
                raw = read_html_first(intl_path)
                intl = raw.copy()
            except Exception:
                pass

    def load_supply(path):
        if not path:
            return pd.DataFrame()
        try:
            d = read_html_first(path)
        except Exception:
            return pd.DataFrame()
        if d.empty:
            return d
        d.columns = [str(c).strip() for c in d.columns]
        return d

    gas = load_supply(gas_path)
    diesel = load_supply(diesel_path)

    retail = pd.DataFrame()
    if retail_path:
        try:
            retail = read_html_first(retail_path)
            retail.columns = [str(c).strip() for c in retail.columns]
        except Exception:
            pass

    frames = []
    if not intl.empty:
        intl = intl.rename(columns={intl.columns[0]: "기간"})
        keep = [c for c in ["기간", "휘발유(92RON)", "경유(0.001%)"] if c in intl.columns]
        x = intl[keep].copy()
        x["date"] = x["기간"].map(parse_week_label)
        for c in keep[1:]:
            x[c] = safe_num(x[c])
        frames.append(x.set_index("date").drop(columns=["기간"]))

    if not gas.empty:
        g = gas.rename(columns={gas.columns[0]: "기간"})
        if "GS칼텍스" in g.columns:
            g["date"] = g["기간"].map(parse_week_label)
            g["GS_휘발유_공급"] = safe_num(g["GS칼텍스"])
            frames.append(g.set_index("date")[["GS_휘발유_공급"]])

    if not diesel.empty:
        d = diesel.rename(columns={diesel.columns[0]: "기간"})
        if "GS칼텍스" in d.columns:
            d["date"] = d["기간"].map(parse_week_label)
            d["GS_경유_공급"] = safe_num(d["GS칼텍스"])
            frames.append(d.set_index("date")[["GS_경유_공급"]])

    if not retail.empty:
        r = retail.rename(columns={retail.columns[0]: "기간"})
        r["date"] = r["기간"].map(parse_week_label)
        for src, dst in [("보통휘발유", "주유소_휘발유_판매"), ("자동차용경유", "주유소_경유_판매")]:
            if src in r.columns:
                r[dst] = safe_num(r[src])
        cols = [c for c in ["주유소_휘발유_판매", "주유소_경유_판매"] if c in r.columns]
        frames.append(r.set_index("date")[cols])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.isna()]
    return out.reset_index()

@st.cache_data
def load_station_events():
    p = find_file("산업통상부_전국 주유소 등록현황_20251231 (1).csv")
    if not p:
        return pd.DataFrame(), pd.DataFrame()
    df = read_csv_kr(p)
    seoul = df[df["소재지"].astype(str).str.startswith("서울")].copy()
    seoul["연도"] = safe_num(seoul["연도"]).astype("Int64")
    seoul["자치구"] = seoul["소재지"].astype(str).str.extract(r"서울(?:특별시)?\s*([가-힣]+구)")
    seoul["영구증감"] = np.select(
        [seoul["구분"].astype(str).eq("신규등록"), seoul["구분"].astype(str).isin(["폐업", "등록취소"])],
        [1, -1], default=0
    )
    yearly = (
        seoul[seoul["구분"].astype(str).isin(["신규등록", "폐업", "등록취소"])]
        .groupby(["연도", "구분"]).size().unstack(fill_value=0).reset_index()
    )
    for c in ["신규등록", "폐업", "등록취소"]:
        if c not in yearly.columns:
            yearly[c] = 0
    yearly["폐업·등록취소"] = yearly["폐업"] + yearly["등록취소"]
    yearly["순증감"] = yearly["신규등록"] - yearly["폐업·등록취소"]

    district = seoul.groupby("자치구", dropna=True).agg(
        신규=("구분", lambda s: int((s == "신규등록").sum())),
        폐업등록취소=("구분", lambda s: int(s.isin(["폐업", "등록취소"]).sum())),
        휴업=("구분", lambda s: int((s == "휴업").sum())),
    ).reset_index()
    district["순증감"] = district["신규"] - district["폐업등록취소"]
    return yearly, district

@st.cache_data
def load_current_station_count():
    p = find_file("한국석유공사_지역별 주유소 수_20251231.csv")
    if not p:
        return np.nan
    df = read_csv_kr(p)
    row = df[df["시도"].astype(str).eq("서울")]
    return float(row["주유소 수"].iloc[0]) if not row.empty else np.nan

@st.cache_data
def load_vehicle_raw():
    p = find_file("서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년7월).xlsx")
    if not p:
        return pd.DataFrame()
    raw = pd.read_excel(p, header=None)
    rows = raw[(raw[0].astype(str).str.startswith("서울특별시 ")) & raw[2].isna() & raw[3].isna()].copy()
    rows["district"] = rows[0].astype(str).str.extract(r"서울특별시\s+([가-힣]+구)")
    rows["vehicles_202607"] = safe_num(rows[4])
    return rows[["district", "vehicles_202607"]].dropna().drop_duplicates("district")

@st.cache_data
def load_district_commercial():
    specs = [
        ("서울시 상권분석서비스(길단위인구-자치구).csv", "총_유동인구_수", "floating"),
        ("서울시 상권분석서비스(상주인구-자치구).csv", "총_상주인구_수", "resident"),
        ("서울시 상권분석서비스(직장인구-자치구).csv", "총_직장_인구_수", "worker"),
    ]
    merged = None
    for filename, value_col, out_col in specs:
        p = find_file(filename)
        if not p:
            continue
        d = latest_quarter(read_csv_kr(p))
        if "자치구_코드_명" not in d.columns or value_col not in d.columns:
            continue
        x = d[["자치구_코드_명", value_col]].copy()
        x.columns = ["district", out_col]
        x[out_col] = safe_num(x[out_col])
        merged = x if merged is None else merged.merge(x, on="district", how="outer")

    ap = find_file("서울시 상권분석서비스(아파트-자치구).csv")
    if ap:
        d = latest_quarter(read_csv_kr(ap))
        hh_cols = [c for c in d.columns if "아파트_면적_" in c and "세대_수" in c]
        if hh_cols:
            d["apt_households"] = d[hh_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        else:
            d["apt_households"] = np.nan
        x = d[["자치구_코드_명", "아파트_단지_수", "apt_households", "아파트_평균_시가"]].copy()
        x.columns = ["district", "apt_complexes", "apt_households", "apt_avg_value"]
        for c in x.columns[1:]:
            x[c] = safe_num(x[c])
        merged = x if merged is None else merged.merge(x, on="district", how="outer")

    return merged if merged is not None else pd.DataFrame()

@st.cache_data
def load_trade_area_data():
    out = {}
    for key, filename in {
        "floating": "서울시 상권분석서비스(길단위인구-상권).csv",
        "resident": "서울시 상권분석서비스(상주인구-상권).csv",
        "worker": "서울시 상권분석서비스(직장인구-상권).csv",
        "apartment": "서울시 상권분석서비스(아파트-상권).csv",
    }.items():
        p = find_file(filename)
        if p:
            try:
                out[key] = latest_quarter(read_csv_kr(p))
            except Exception:
                out[key] = pd.DataFrame()
        else:
            out[key] = pd.DataFrame()
    return out

price_weekly = load_price_cap_data()
station_yearly, station_district = load_station_events()
seoul_station_count_2025 = load_current_station_count()
vehicle_raw = load_vehicle_raw()
district_commercial = load_district_commercial()
trade_area = load_trade_area_data()

# =========================================================
# Previous normalized analysis helpers
# =========================================================
def district_vehicle_series(district):
    if veh.empty:
        return pd.DataFrame(columns=["month", "count"])
    d = veh[(veh["sido"] == "서울") & (veh["sigungu"] == district)].copy()
    if d.empty:
        return pd.DataFrame(columns=["month", "count"])
    if "vehicle_type" in d.columns and (d["vehicle_type"] == "총계").any():
        d = d[d["vehicle_type"] == "총계"]
    return d.groupby("month", as_index=False)["count"].sum().sort_values("month")

def latest_vehicle(district):
    d = district_vehicle_series(district)
    if not d.empty:
        latest = d.iloc[-1]
        latest_n = float(latest["count"])
        prev = d[d["month"] == latest["month"] - pd.DateOffset(years=1)]
        yoy = np.nan if prev.empty or float(prev.iloc[0]["count"]) == 0 else (latest_n / float(prev.iloc[0]["count"]) - 1) * 100
        return latest_n, yoy
    if not vehicle_raw.empty:
        x = vehicle_raw[vehicle_raw["district"] == district]
        if not x.empty:
            return float(x.iloc[0]["vehicles_202607"]), np.nan
    return np.nan, np.nan

def network_summary():
    if net.empty:
        base = pd.DataFrame({"district": TARGET_DISTRICTS})
        if not vehicle_raw.empty:
            base = base.merge(vehicle_raw, on="district", how="left").rename(columns={"vehicles_202607":"vehicles"})
        else:
            base["vehicles"] = np.nan
        if not station_district.empty:
            base = base.merge(station_district, left_on="district", right_on="자치구", how="left").drop(columns=["자치구"], errors="ignore")
        return base

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

    share_low = pct_rank_low(out["gs_share"])
    vps_high = pct_rank_high(out["vehicles_per_station"])
    yoy_fill = out["vehicle_yoy"].fillna(out["vehicle_yoy"].median())
    yoy_high = pct_rank_high(yoy_fill)
    out["expansion_score"] = (share_low + vps_high + yoy_high) / 3
    share_high = pct_rank_high(out["gs_share"])
    vps_low = pct_rank_low(out["vehicles_per_station"])
    out["efficiency_score"] = (share_high + vps_low) / 2
    out["price_watch_score"] = pct_rank_high(out["avg_price_gap"].fillna(out["avg_price_gap"].median()))

    if not station_district.empty:
        out = out.merge(station_district, left_on="district", right_on="자치구", how="left").drop(columns=["자치구"], errors="ignore")
    return out

summary = network_summary()

# =========================================================
# Opinet API - existing ancillary facilities are top priority
# =========================================================
def get_api_key():
    secret_key = ""
    try:
        secret_key = st.secrets.get("OPINET_API_KEY", "")
    except Exception:
        secret_key = ""

    entered = st.sidebar.text_input(
        "오피넷 API 인증키",
        value=secret_key,
        type="password",
        help="오피넷에서 발급받은 인증키입니다. Streamlit Secrets의 OPINET_API_KEY로 넣어도 됩니다.",
        key="opinet_api_key_input",
    )
    return entered.strip()

def opinet_detail(key, station_id):
    url = "https://www.opinet.co.kr/api/detailById.do"
    params = {"certkey": key, "out": "json", "id": station_id}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    js = r.json()
    result = js.get("RESULT", {})
    oil = result.get("OIL", [])
    if isinstance(oil, list):
        return oil[0] if oil else {}
    if isinstance(oil, dict):
        return oil
    return {}

def detect_station_id_col(df):
    for c in ["station_id", "UNI_ID", "uni_id", "id", "os_id"]:
        if c in df.columns:
            return c
    return None

def yn_text(v):
    s = str(v).strip().upper()
    return "있음" if s in ["Y", "1", "TRUE"] else "없음" if s in ["N", "0", "FALSE"] else "확인 불가"

# =========================================================
# Header
# =========================================================
API_KEY = get_api_key()

st.title("⛽ GS칼텍스 국내영업 Network·유외수익 의사결정 지원 도구")
st.caption("최고가격제 → 서울 Network 축소 → 지역 수요 → GS Network → 기존 유외시설 → 저투자형 수익기회 순으로 분석")

if API_KEY:
    st.success("오피넷 API 연결 준비 완료 · 개별 GS 주유소의 세차장/경정비/편의점 정보를 조회할 수 있습니다.")
else:
    st.warning(
        "오피넷 API 인증키가 아직 입력되지 않았습니다. "
        "네가 준 가격·주유소변동·자동차·상권 원자료 분석은 그대로 작동하지만, "
        "GS 주유소별 기존 유외시설(CAR_WASH_YN / MAINT_YN / CVS_YN) 조회는 API 연결 후 활성화됩니다."
    )

with st.expander("현재 원자료 연결 상태", expanded=False):
    source_checks = {
        "국제 석유제품가격": find_file("국제_석유제품가격2026011-2026092.xlsx") or find_file("국제_석유제품가격2026011-2026092.xls"),
        "정유사 주간 공급가격": find_file("정유사_주간공급가격_회사별.xls"),
        "주유소 평균판매가격": find_file("주유소_제품별_평균판매가격.xls"),
        "전국 주유소 등록현황": find_file("산업통상부_전국 주유소 등록현황_20251231 (1).csv"),
        "서울 자동차 등록현황": find_file("서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년7월).xlsx"),
        "길단위인구-상권": find_file("서울시 상권분석서비스(길단위인구-상권).csv"),
        "상주인구-상권": find_file("서울시 상권분석서비스(상주인구-상권).csv"),
        "직장인구-상권": find_file("서울시 상권분석서비스(직장인구-상권).csv"),
        "아파트-상권": find_file("서울시 상권분석서비스(아파트-상권).csv"),
        "점포-상권": find_file("서울시 상권분석서비스(점포-상권)_2025년.zip"),
        "추정매출-상권": find_file("서울시 상권분석서비스(추정매출-상권)_2025년.zip"),
    }
    status_df = pd.DataFrame(
        [{"자료": k, "상태": "연결됨" if v else "확인 필요", "파일": Path(v).name if v else "-"}
         for k, v in source_checks.items()]
    )
    st.dataframe(status_df, hide_index=True, use_container_width=True)
    st.caption("GS 주유소별 세차장·경정비·편의점은 파일이 아니라 오피넷 API에서 조회합니다.")

with st.expander("분석 원칙과 공개자료 한계", expanded=False):
    st.markdown("""
- **최고가격제 탭**은 국제 제품가격, GS칼텍스 주간 공급가격, 주유소 평균 판매가격을 구분해 봅니다. `주유소 판매가격 - 정유사 공급가격`을 주유소 마진으로 계산하지 않습니다.
- **Network 변화 탭**은 신규등록과 폐업·등록취소만 영구 증감으로 보고, 휴업은 별도 표시합니다.
- **DC/AC**는 공개 오피넷 자료만으로 개별 유형을 식별할 수 없어 임의 분류하지 않습니다.
- **유외수익 분석의 우선순위는 신규 CAPEX가 아니라 기존 시설 활용**입니다. 오피넷 상세 API의 `CAR_WASH_YN`, `MAINT_YN`, `CVS_YN`을 가장 먼저 확인합니다.
- 상권자료는 현재 자치구/상권 단위 수요 특성을 보여주지만, 상권 경계좌표가 없으므로 **개별 주유소와 상권을 임의로 공간 매칭하지 않습니다.**
- 공개자료로 확인할 수 없는 실제 판매량, 공급마진, 임차료, 운영비, 시설가동률, 계약조건은 별도 내부 확인 항목으로 남깁니다.
""")

tabs = st.tabs([
    "① 분석 개요",
    "② 최고가격제 검증",
    "③ 서울 Network 변화",
    "④ 수요·시장",
    "⑤ GS Network·가격",
    "⑥ 유외수익 기회",
    "⑦ 개별 GS 주유소",
    "⑧ 최종결론",
])

# =========================================================
# 1 Overview
# =========================================================
with tabs[0]:
    st.subheader("분석의 출발점")
    st.markdown("""
<div class="section-text">
이 프로그램은 <b>“직접 보유 CC를 줄이고 국내 공급가격 운용에도 제약이 있는 상황에서, 어떻게 기존 Network의 물량과 수익기반을 지킬 것인가?”</b>라는 질문에서 출발합니다.<br><br>
첫째, 2026년 국제 석유제품가격과 GS칼텍스 공급가격을 비교해 최고가격제 시행 이후 가격 전가가 실제로 제약됐는지를 확인합니다. 둘째, 서울 주유소 신규·폐업 이력으로 물리적 Network가 장기간 축소되고 있는지를 봅니다. 셋째, 자동차 등록대수와 GS Network 비중을 연결해 기존 Network의 중요성이 큰 지역을 선별합니다.<br><br>
그 다음부터는 <b>“가격지원이나 신규 시설투자를 더 하자”</b>가 아니라 <b>“현재 남아 있는 DC·AC 등 Network의 사업자 수익성을 어떻게 저투자로 높일 것인가”</b>를 봅니다. 따라서 개별 GS 주유소에서 가장 먼저 세차장·경정비·편의점 등 기존 유외시설 보유 여부를 확인하고, 지역의 자동차·유동·상주·직장·아파트 수요를 연결해 기존 시설 활용 또는 파트너형 서비스를 검토합니다.
</div>
""", unsafe_allow_html=True)

    flow = pd.DataFrame({
        "단계": ["1. 가격 제약", "2. Network 변화", "3. 지역 수요", "4. GS 침투도", "5. 기존 유외시설", "6. 상권 수요", "7. 영업 제안"],
        "확인 질문": [
            "국제가격 상승을 국내 공급가격에 충분히 반영할 수 있었나?",
            "서울 주유소 Network는 실제로 줄고 있나?",
            "어느 지역의 자동차 기반 수요가 큰가?",
            "수요 대비 GS Network가 충분한가?",
            "세차·경정비·편의점 등 이미 가진 자산은 무엇인가?",
            "주거·직장·유동인구 특성상 어떤 서비스가 맞는가?",
            "추가 CAPEX를 최소화하면서 무엇부터 실행할 것인가?",
        ],
        "결론 형태": [
            "가격 전가 제약 확인",
            "기존 Network 유지 중요성",
            "우선 점검지역",
            "Network 확보/물량방어",
            "신규투자보다 활용 우선",
            "저투자형 유외수익 후보",
            "현장 추가확인 항목",
        ]
    })
    st.dataframe(flow, hide_index=True, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("가격 주간데이터", f"{len(price_weekly):,}주" if not price_weekly.empty else "파일 필요")
    c2.metric("서울 2025년말 주유소", "-" if pd.isna(seoul_station_count_2025) else f"{seoul_station_count_2025:,.0f}개")
    c3.metric("자동차 원자료", "2026.07" if not vehicle_raw.empty else "파일 필요")
    c4.metric("상권 수요자료", "연결 완료" if not district_commercial.empty else "파일 필요")

# =========================================================
# 2 Price cap
# =========================================================
with tabs[1]:
    st.subheader("최고가격제 시행 전후 가격 전가 제약")
    guide(
        "① 유외사업 논리의 첫 번째 전제를 데이터로 확인합니다.",
        "정유사의 공급가격 운용이 제약됐다면 DC·AC 경쟁력을 가격지원만으로 높이는 전략에도 한계가 생길 수 있습니다.",
        "국제제품가격(USD/bbl), GS칼텍스 공급가격(원/L), 전국 주유소 평균 판매가격(원/L)을 각각 분리해 보고, 변동률 비교에는 2026년 1월 1주=100 지수를 사용합니다.",
        "국제가격 급등기에 GS 공급가격이 동일한 폭으로 움직이지 않았는지 확인합니다. 다만 이를 리터당 마진 감소액으로 환산하지 않습니다.",
    )
    if price_weekly.empty:
        st.warning("가격 원자료를 찾지 못했습니다.")
    else:
        fuel = st.radio("제품", ["휘발유", "경유"], horizontal=True, key="cap_fuel")
        if fuel == "휘발유":
            intl_col, supply_col, retail_col = "휘발유(92RON)", "GS_휘발유_공급", "주유소_휘발유_판매"
        else:
            intl_col, supply_col, retail_col = "경유(0.001%)", "GS_경유_공급", "주유소_경유_판매"

        d = price_weekly[["date", intl_col, supply_col, retail_col]].dropna(how="all", subset=[intl_col, supply_col, retail_col]).copy()
        base = d[[intl_col, supply_col, retail_col]].apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
        for c in [intl_col, supply_col, retail_col]:
            d[c + "_idx"] = d[c] / base[c] * 100 if pd.notna(base[c]) and base[c] != 0 else np.nan

        idx = d.melt("date", value_vars=[intl_col+"_idx", supply_col+"_idx", retail_col+"_idx"], var_name="series", value_name="index")
        idx["series"] = idx["series"].map({
            intl_col+"_idx":"국제제품가격 지수",
            supply_col+"_idx":"GS 공급가격 지수",
            retail_col+"_idx":"주유소 평균판매가격 지수",
        })
        fig = px.line(idx, x="date", y="index", color="series", markers=True,
                      labels={"date":"주간 대표일", "index":"2026년 1월 1주=100", "series":""},
                      title=f"{fuel} 가격 변동률 비교")
        fig.add_vline(x=pd.Timestamp("2026-03-13").timestamp()*1000, line_dash="dash", annotation_text="최고가격제 시행")
        fig.update_layout(height=430, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)

        absdf = d.melt("date", value_vars=[supply_col, retail_col], var_name="series", value_name="price")
        absdf["series"] = absdf["series"].map({supply_col:"GS 공급가격", retail_col:"주유소 평균판매가격"})
        fig2 = px.line(absdf, x="date", y="price", color="series", markers=True,
                       labels={"date":"주간 대표일", "price":"원/L", "series":""},
                       title=f"{fuel} 국내 가격 흐름")
        fig2.add_vline(x=pd.Timestamp("2026-03-13").timestamp()*1000, line_dash="dash", annotation_text="3/13 시행")
        fig2.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        # verified comparison window
        pre = d[d["date"] <= pd.Timestamp("2026-02-22")].tail(1)
        peak = d[(d["date"] >= pd.Timestamp("2026-03-15")) & (d["date"] <= pd.Timestamp("2026-04-05"))]
        if not pre.empty and not peak.empty:
            peak_i = peak[intl_col].idxmax()
            b = pre.iloc[0]
            p = d.loc[peak_i]
            intl_change = (p[intl_col] / b[intl_col] - 1) * 100 if pd.notna(b[intl_col]) else np.nan
            supply_change = (p[supply_col] / b[supply_col] - 1) * 100 if pd.notna(b[supply_col]) else np.nan
            c1,c2,c3 = st.columns(3)
            c1.metric("비교 기준 국제가격", f"{b[intl_col]:,.2f} USD/bbl")
            c2.metric("국제가격 변화", f"{intl_change:+.1f}%")
            c3.metric("GS 공급가격 변화", f"{supply_change:+.1f}%")

        conclusion_box([
            "국제제품가격과 국내 공급가격은 단위가 다르므로 절대값 차이를 마진으로 계산하지 않고, 동일 기간의 변동 방향과 지수만 비교합니다.",
            "2026년 3월 국제제품가격 급등기에 GS칼텍스 공급가격은 국제가격과 같은 폭으로 움직이지 않았고, 시행 이후 공급가격 운용 제약이 실제 데이터에서 확인됩니다.",
            "주유소 판매가격은 최고가격제의 직접 규제대상이 아니므로 공급가격과 다르게 움직일 수 있습니다. 따라서 세 가격을 반드시 분리해서 봅니다.",
            "이 결과는 '유외사업이 무조건 필요하다'는 결론이 아니라, DC·AC 경쟁력을 가격지원만으로 높이기 어려울 수 있으므로 저투자형 추가 수익원을 검토할 근거가 됩니다.",
        ])

# =========================================================
# 3 Network changes
# =========================================================
with tabs[2]:
    st.subheader("서울 주유소 Network 장기 축소")
    guide(
        "② 실제로 물리적 Network가 줄고 있는지 검증합니다.",
        "신규출점이 거의 없는 시장이라면 기존 사업자 유지와 Network당 생산성이 더 중요해집니다.",
        "2015~2025년 서울의 신규등록, 폐업·등록취소를 영구 증감으로 계산하고 휴업은 별도로 봅니다.",
        "서울 전체 및 자치구별로 Network 축소 강도가 다른지 확인합니다.",
    )
    if station_yearly.empty:
        st.warning("주유소 등록현황 원자료를 찾지 못했습니다.")
    else:
        y = station_yearly.copy()
        fig = go.Figure()
        fig.add_bar(x=y["연도"], y=y["신규등록"], name="신규등록")
        fig.add_bar(x=y["연도"], y=-y["폐업·등록취소"], name="폐업·등록취소")
        fig.add_scatter(x=y["연도"], y=y["순증감"], name="순증감", mode="lines+markers")
        fig.update_layout(title="서울 주유소 신규등록 vs 폐업·등록취소", barmode="relative", height=430,
                          xaxis_title="", yaxis_title="건", margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)

        total_new = int(y["신규등록"].sum())
        total_out = int(y["폐업·등록취소"].sum())
        total_net = int(y["순증감"].sum())
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("2015~2025 신규등록", f"{total_new:,}건")
        c2.metric("폐업·등록취소", f"{total_out:,}건")
        c3.metric("순증감", f"{total_net:+,}개")
        c4.metric("2025년말 서울 주유소", "-" if pd.isna(seoul_station_count_2025) else f"{seoul_station_count_2025:,.0f}개")

        d = station_district.sort_values("순증감").copy()
        fig2 = px.bar(d, x="자치구", y="순증감", title="자치구별 2015~2025 주유소 순증감",
                      labels={"자치구":"","순증감":"개"})
        fig2.update_layout(height=420, xaxis_tickangle=-45, margin=dict(l=10,r=10,t=55,b=80))
        st.plotly_chart(fig2, use_container_width=True)

        dd = st.selectbox("자치구별 결론", TARGET_DISTRICTS, key="event_district")
        row = d[d["자치구"] == dd]
        if not row.empty:
            r = row.iloc[0]
            conclusion_box([
                f"{dd}에서는 2015~2025년 신규등록 {int(r['신규'])}건, 폐업·등록취소 {int(r['폐업등록취소'])}건으로 영구 Network 순증감은 {int(r['순증감']):+d}개입니다.",
                f"휴업은 {int(r['휴업'])}건이지만 재영업 가능성이 있으므로 영구 감소에 포함하지 않았습니다.",
                "서울 전체적으로 신규등록보다 폐업·등록취소가 훨씬 많아 '신규 주유소를 계속 늘리는 시장'보다는 '기존 Network의 유지와 생산성을 관리해야 하는 시장'에 가깝습니다.",
                "다만 폐업 원인을 이 자료만으로 토지가격·수익성·재개발 등 특정 요인으로 단정하지 않습니다.",
            ])

# =========================================================
# 4 Demand / market
# =========================================================
with tabs[3]:
    st.subheader("자동차 기반 수요와 Network 감소를 함께 보기")
    guide(
        "③ Network 감소가 큰 지역 중에서도 수요가 큰 곳을 구분합니다.",
        "주유소가 많이 줄었다는 사실만으로 영업 우선지역이 되지는 않습니다.",
        "2026년 7월 자동차 등록대수와 2015~2025년 주유소 순증감을 결합합니다.",
        "자동차 수요가 큰데 Network가 많이 감소한 지역은 기존 Network 물량방어와 생산성 제고를 우선 확인할 후보가 됩니다.",
    )
    if vehicle_raw.empty or station_district.empty:
        st.warning("자동차 또는 주유소 변동 원자료를 찾지 못했습니다.")
    else:
        dm = vehicle_raw.merge(station_district, left_on="district", right_on="자치구", how="left").drop(columns=["자치구"], errors="ignore")
        dm["폐업등록취소"] = dm["폐업등록취소"].fillna(0)
        fig = px.scatter(dm, x="vehicles_202607", y="순증감", text="district", size="폐업등록취소",
                         labels={"vehicles_202607":"2026.07 자동차 등록대수", "순증감":"2015~2025 주유소 순증감"},
                         title="현재 자동차 수요 vs 과거 주유소 Network 변화")
        fig.update_traces(textposition="top center")
        fig.add_vline(x=dm["vehicles_202607"].median(), line_dash="dot")
        fig.add_hline(y=dm["순증감"].median(), line_dash="dot")
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)

        high_demand = dm["vehicles_202607"] >= dm["vehicles_202607"].median()
        large_decline = dm["순증감"] <= dm["순증감"].median()
        cand = dm[high_demand & large_decline].sort_values(["vehicles_202607","순증감"], ascending=[False,True])
        st.markdown("#### 고수요 + Network 감소폭 큰 지역")
        show = cand[["district","vehicles_202607","신규","폐업등록취소","순증감"]].copy()
        show.columns = ["자치구","자동차 등록대수","신규","폐업·등록취소","Network 순증감"]
        st.dataframe(show.style.format({"자동차 등록대수":"{:,.0f}","신규":"{:,.0f}","폐업·등록취소":"{:,.0f}","Network 순증감":"{:+.0f}"}), hide_index=True, use_container_width=True)

        conclusion_box([
            "자동차가 많은 지역에서도 주유소 Network는 크게 줄어들 수 있어 '자동차가 많다 = 신규출점'으로 연결하면 안 됩니다.",
            "고수요·Network 감소 지역에서는 남아 있는 기존 Network 한 곳의 중요성이 상대적으로 커질 수 있으므로 DC·AC 유지, 판매량, 계열이탈 위험을 우선 점검할 근거가 됩니다.",
            "반대로 수요가 낮은 지역의 Network 감소는 기존점 효율성을 먼저 확인해야 하며, 단순 보충출점 논리로 접근하지 않습니다.",
            "다음 단계에서는 GS Network 비중과 개별 점포의 기존 유외시설을 연결해 '신규투자 없이 무엇을 활용할지'를 봅니다.",
        ])

    if CORE_OK:
        st.divider()
        st.markdown("### 기존 프로그램의 서울 석유수요·Network 밀도")
        fuel = st.radio("제품", ["휘발유","경유"], horizontal=True, key="demand_fuel")
        volume_col, _, _ = FUEL_COLS[fuel]
        d = cons[cons["region"]=="서울"].sort_values("month").copy()
        if not d.empty and volume_col in d.columns:
            fig = px.line(d, x="month", y=volume_col, markers=True, title=f"서울 {fuel} 월별 소비량",
                          labels={"month":"월", volume_col:"천 Bbl"})
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 5 GS Network / price
# =========================================================
with tabs[4]:
    st.subheader("GS Network 침투도와 소비자가격 포지션")
    if not CORE_OK:
        st.info("이 탭의 브랜드별 현재 Network 비교는 오피넷 실시간 Network 데이터 연결이 필요한 영역입니다. 네가 준 원자료가 누락된 것이 아닙니다.")
    else:
        guide(
            "④ 지역별 GS Network의 상대적 위치를 확인합니다.",
            "고수요 지역에서도 GS Network가 이미 충분한 곳과 부족한 곳은 대응이 달라야 합니다.",
            "전체 주유소 수, GS 주유소 수, GS Network 비중, 주유소당 차량, 자치구 평균 대비 GS 소비자가격을 연결합니다.",
            "Network 확보 우선인지, 기존 Network 물량방어 우선인지, 가격·거래조건 추가점검이 필요한지 구분합니다.",
        )
        cnt = net.groupby(["district","brand"]).size().reset_index(name="주유소수")
        fig = px.bar(cnt, x="district", y="주유소수", color="brand", barmode="stack", title="자치구별 브랜드 Network 구성")
        fig.update_layout(height=450, xaxis_tickangle=-45, xaxis_title="", margin=dict(l=10,r=10,t=55,b=80))
        st.plotly_chart(fig, use_container_width=True)

        district = st.selectbox("자치구", TARGET_DISTRICTS, key="network_price_district")
        r = summary[summary["district"]==district].iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("전체 주유소", f"{r['stations']:,.0f}개")
        c2.metric("GS 주유소", f"{r['gs_stations']:,.0f}개")
        c3.metric("GS Network 비중", f"{r['gs_share']:.1f}%")
        c4.metric("주유소당 등록차량", f"{r['vehicles_per_station']:,.0f}대")

        fuel = st.radio("가격 제품", ["휘발유","경유"], horizontal=True, key="price_fuel")
        _, price_col, _ = FUEL_COLS[fuel]
        dd = net[net["district"]==district].dropna(subset=[price_col]).copy()
        stats = dd.groupby("brand")[price_col].agg(["count","min","median","mean","max"]).reset_index().sort_values("mean")
        stats.columns = ["브랜드","주유소 수","최저","중앙","평균","최고"]
        st.dataframe(stats.style.format({"주유소 수":"{:,.0f}","최저":"{:,.0f}","중앙":"{:,.0f}","평균":"{:,.0f}","최고":"{:,.0f}"}), hide_index=True, use_container_width=True)

        gsavg = dd.loc[dd["brand"]=="GS칼텍스", price_col].mean()
        allavg = dd[price_col].mean()
        gap = gsavg - allavg
        conclusion_box([
            f"{district}의 GS Network 비중은 {r['gs_share']:.1f}%이고 주유소당 등록차량은 {r['vehicles_per_station']:,.0f}대입니다.",
            f"{fuel} 소비자가격은 GS 평균이 자치구 전체 평균보다 {gap:+,.0f}원/L입니다." if pd.notna(gap) else "GS 평균가격을 계산할 수 없습니다.",
            "소비자가격 프리미엄은 정유사 공급마진을 뜻하지 않으며, 입지·서비스·판촉·경쟁점 가격을 추가로 확인하는 신호로만 사용합니다.",
            "Network 확대·축소가 아니라 DC·AC 등 기존/신규 Network 확보와 기존점 물량방어 중 무엇을 먼저 볼지 정하는 단계입니다.",
        ])

# =========================================================
# 6 Non-fuel opportunity
# =========================================================
with tabs[5]:
    st.subheader("유외수익 기회 · 기존 자산 활용이 먼저")
    guide(
        "⑤ 가격지원보다 먼저 기존 유외시설과 지역 수요를 확인합니다.",
        "최고가격제로 정유사 공급가격 운용이 제약되는 환경에서 추가적인 현금성 지원이나 신규 CAPEX 확대는 부담이 될 수 있습니다.",
        "자치구별 자동차·유동·상주·직장·아파트 수요를 비교하고, 개별 GS 주유소에서는 오피넷 상세 API의 세차장·경정비·편의점 보유 여부를 가장 먼저 확인합니다.",
        "기존 시설 가동률 개선 → 기존 공간의 파트너형 활용 → 추가투자 검토 순서로 봅니다.",
    )

    if district_commercial.empty:
        st.warning("서울시 상권분석 자치구 자료를 찾지 못했습니다.")
    else:
        d = district_commercial.merge(vehicle_raw, on="district", how="left") if not vehicle_raw.empty else district_commercial.copy()
        for c in ["floating","resident","worker","apt_households","vehicles_202607"]:
            if c in d.columns:
                d[c+"_pct"] = pct_rank_high(d[c])

        weights = [c for c in ["floating_pct","resident_pct","worker_pct","apt_households_pct","vehicles_202607_pct"] if c in d.columns]
        d["nonfuel_demand_score"] = d[weights].mean(axis=1) if weights else np.nan

        st.markdown("#### 자치구별 유외수익 수요환경")
        cols = ["district"] + [c for c in ["vehicles_202607","floating","resident","worker","apt_households","nonfuel_demand_score"] if c in d.columns]
        show = d[cols].sort_values("nonfuel_demand_score", ascending=False)
        rename = {
            "district":"자치구","vehicles_202607":"자동차","floating":"길단위인구","resident":"상주인구",
            "worker":"직장인구","apt_households":"아파트 세대수","nonfuel_demand_score":"유외수요 스크리닝"
        }
        st.dataframe(show.rename(columns=rename).style.format({
            "자동차":"{:,.0f}","길단위인구":"{:,.0f}","상주인구":"{:,.0f}","직장인구":"{:,.0f}",
            "아파트 세대수":"{:,.0f}","유외수요 스크리닝":"{:.0f}"
        }), hide_index=True, use_container_width=True, height=470)

        district = st.selectbox("자치구별 유외전략 보기", TARGET_DISTRICTS, key="nonfuel_district")
        r = d[d["district"]==district]
        if not r.empty:
            r = r.iloc[0]
            vehicle_hi = r.get("vehicles_202607_pct", 0) >= 60
            resident_hi = r.get("resident_pct", 0) >= 60
            worker_hi = r.get("worker_pct", 0) >= 60
            floating_hi = r.get("floating_pct", 0) >= 60
            apt_hi = r.get("apt_households_pct", 0) >= 60

            ideas = []
            if vehicle_hi:
                ideas.append("기존 세차·경정비 시설이 있다면 신규설비보다 가동률·교차판매 개선을 우선 검토")
            if resident_hi or apt_hi:
                ideas.append("주거·아파트 수요가 높다면 기존 유휴공간을 활용한 픽업·생활물류 파트너십 검토")
            if worker_hi and floating_hi:
                ideas.append("직장·유동수요가 모두 높다면 출퇴근·업무동선형 편의·픽업 서비스를 검토")
            if not ideas:
                ideas.append("공개 수요지표만으로 강한 유외수요 신호가 없어 신규투자보다 기존 시설 실적 확인을 우선")

            st.markdown("#### 이 지역에서 먼저 볼 것")
            for x in ideas:
                st.write("• " + x)

            conclusion_box([
                f"{district}의 유외수요 스크리닝은 {r['nonfuel_demand_score']:.0f}점입니다. 이 점수는 매출예측이 아니라 지역 수요특성의 상대순위입니다.",
                "가장 먼저 해야 할 일은 해당 지역 GS 주유소의 세차장·경정비·편의점 보유 여부와 실제 이용률을 확인하는 것입니다.",
                "시설이 이미 있다면 신규 CAPEX보다 가동률·동선·예약/결제·교차판매를 개선하고, 시설이 없다면 바로 신설하지 않고 외부 파트너가 비용을 부담하는 공간활용 모델부터 검토합니다.",
                "상권분석 데이터는 수요의 방향을 보여줄 뿐 개별 주유소의 임대가능면적·운영비·매출을 알 수 없으므로 최종투자 판단에는 사용하지 않습니다.",
            ])

        st.markdown("#### 상권 단위 수요 탐색 · 개별 주유소와 자동 매칭하지 않음")
        if all(k in trade_area and not trade_area[k].empty for k in ["floating","resident","worker"]):
            names = sorted(set(trade_area["floating"]["상권_코드_명"].dropna().astype(str)))
            ta = st.selectbox("상권 선택", names, key="trade_area")
            def val(df, col):
                x = df[df["상권_코드_명"].astype(str)==ta]
                return safe_num(x[col]).iloc[0] if not x.empty and col in x.columns else np.nan
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("길단위인구", fmt_num(val(trade_area["floating"], "총_유동인구_수")))
            c2.metric("상주인구", fmt_num(val(trade_area["resident"], "총_상주인구_수")))
            c3.metric("직장인구", fmt_num(val(trade_area["worker"], "총_직장_인구_수")))
            apt = val(trade_area.get("apartment", pd.DataFrame()), "아파트_단지_수")
            c4.metric("아파트 단지", fmt_num(apt))
            st.caption("현재 확보한 파일에는 상권 경계좌표가 없어 선택한 GS 주유소와 이 상권을 임의로 연결하지 않습니다.")

# =========================================================
# 7 Individual station / facility
# =========================================================
with tabs[6]:
    st.subheader("개별 GS 주유소 · 기존 유외시설부터 확인")
    if not CORE_OK:
        st.info("개별 GS 주유소 분석은 오피넷의 주유소 ID/Network 목록을 연결한 뒤 활성화됩니다. 네가 준 원자료가 빠진 것이 아닙니다.")
    else:
        gs = net[net["brand"]=="GS칼텍스"].copy()
        if gs.empty:
            st.warning("GS 주유소 데이터가 없습니다.")
        else:
            district = st.selectbox("자치구", sorted(gs["district"].dropna().unique()), key="facility_district")
            g = gs[gs["district"]==district].sort_values("station_name")
            station = st.selectbox("GS칼텍스 주유소", g["station_name"].tolist(), key="facility_station")
            row = g[g["station_name"]==station].iloc[0]
            st.write(f"주소: {row.get('address','-')}")

            id_col = detect_station_id_col(g)
            api_key = API_KEY
            st.markdown("#### 1순위 확인: 기존 유외시설")
            if id_col is None:
                st.warning("현재 network_snapshot.csv에 오피넷 주유소 ID가 없어 상세시설 API를 자동 호출할 수 없습니다. `station_id` 또는 `UNI_ID` 열을 저장하면 자동 연결됩니다.")
            elif not api_key:
                st.info("Streamlit Secrets에 `OPINET_API_KEY`를 넣으면 세차장·경정비·편의점 보유 여부를 자동 조회합니다.")
                st.code(f"선택 주유소 ID: {row[id_col]}")
            else:
                try:
                    detail = opinet_detail(api_key, row[id_col])
                    c1,c2,c3 = st.columns(3)
                    c1.metric("세차장", yn_text(detail.get("CAR_WASH_YN")))
                    c2.metric("경정비", yn_text(detail.get("MAINT_YN")))
                    c3.metric("편의점", yn_text(detail.get("CVS_YN")))
                    existing = [
                        name for name,key in [("세차장","CAR_WASH_YN"),("경정비","MAINT_YN"),("편의점","CVS_YN")]
                        if yn_text(detail.get(key))=="있음"
                    ]
                    if existing:
                        st.success("기존 시설 활용 우선: " + ", ".join(existing))
                    else:
                        st.info("확인된 기존 유외시설이 없습니다. 곧바로 신규투자를 권하지 않고 공간·수요·파트너 비용분담 가능성을 추가 확인합니다.")
                except Exception as e:
                    st.warning("오피넷 상세시설 조회를 완료하지 못했습니다. API Key/주유소 ID를 확인하세요.")

            st.markdown("#### 가격 위치")
            fuel = st.radio("제품", ["휘발유","경유"], horizontal=True, key="station_price_fuel")
            _, price_col, _ = FUEL_COLS[fuel]
            dd = net[net["district"]==district].dropna(subset=[price_col])
            sp = row.get(price_col, np.nan)
            davg = dd[price_col].mean()
            gsavg = dd.loc[dd["brand"]=="GS칼텍스", price_col].mean()
            c1,c2,c3 = st.columns(3)
            c1.metric("선택 주유소", "-" if pd.isna(sp) else f"{sp:,.0f}원/L")
            c2.metric("자치구 평균 대비", "-" if pd.isna(sp) else f"{sp-davg:+,.0f}원/L")
            c3.metric("GS 평균 대비", "-" if pd.isna(sp) else f"{sp-gsavg:+,.0f}원/L")

            st.markdown("#### 현장에서 추가로 확인할 데이터")
            st.dataframe(pd.DataFrame({
                "항목":["기존 시설 이용건수","유외매출·마진","유휴공간/동선","시설 운영비","주변 동일서비스 경쟁","DC·AC 계약/계열이탈 위험"],
                "왜 필요한가":[
                    "시설이 있어도 실제 활용도가 낮을 수 있음",
                    "유류마진 의존도 완화 효과 확인",
                    "신규 CAPEX 없이 파트너 서비스를 넣을 수 있는지 확인",
                    "추가수익보다 운영비가 큰지 확인",
                    "세차·편의·픽업 포화 여부 확인",
                    "유외수익 개선이 Network 유지에 실제 도움이 되는지 확인",
                ]
            }), hide_index=True, use_container_width=True)

# =========================================================
# 8 Final
# =========================================================
with tabs[7]:
    st.subheader("최종결론 · 유실물량 방어 + 저투자형 Network 수익성 개선")

    st.markdown("### 1. 지금까지 데이터로 확인된 사실")
    facts = []
    if not price_weekly.empty:
        facts.append("2026년 3월 국제 석유제품가격 급등기에 GS칼텍스 공급가격은 동일한 폭으로 움직이지 않아 가격 전가 제약이 관찰됩니다.")
    if not station_yearly.empty:
        facts.append(f"2015~2025년 서울의 영구 주유소 순증감은 {int(station_yearly['순증감'].sum()):+d}개로, 신규보다 폐업·등록취소가 훨씬 많았습니다.")
    if not vehicle_raw.empty and not station_district.empty:
        facts.append("자동차 등록대수가 큰 자치구에서도 Network 감소가 크게 나타나는 곳이 있어, 수요가 크다는 이유만으로 신규출점을 결론낼 수 없습니다.")
    facts.append("따라서 핵심 질문은 '주유소를 더 지을까'보다 '남아 있는 DC·AC 등 Network의 경쟁력과 사업자 수익성을 어떻게 지킬까'에 가깝습니다.")
    for i,f in enumerate(facts,1):
        st.write(f"{i}. {f}")

    st.markdown("### 2. 영업전략 우선순위")
    strategy = pd.DataFrame({
        "우선순위":["1순위","2순위","3순위","4순위","후순위"],
        "전략":["기존 유외시설 활용","기존 공간 + 외부 파트너","DC·AC 거래조건/계열이탈 방어","알뜰·고속도로·공공입찰 물량","신규 CAPEX형 유외시설"],
        "판단 기준":[
            "세차·경정비·편의점 보유 + 지역 수요",
            "주거/직장/유동수요 + 유휴공간 + 파트너 비용분담",
            "판매량·지원비·여신·물류·계약조건",
            "물량뿐 아니라 입찰마진·정산·물류 리스크",
            "기존 시설/파트너형 대안으로 부족할 때만 검토",
        ]
    })
    st.dataframe(strategy, hide_index=True, use_container_width=True)

    if not district_commercial.empty and not vehicle_raw.empty:
        d = district_commercial.merge(vehicle_raw, on="district", how="left")
        for c in ["floating","resident","worker","apt_households","vehicles_202607"]:
            if c in d.columns:
                d[c+"_pct"] = pct_rank_high(d[c])
        score_cols = [c for c in d.columns if c.endswith("_pct")]
        d["nonfuel_demand_score"] = d[score_cols].mean(axis=1)
        if not station_district.empty:
            d = d.merge(station_district[["자치구","순증감"]], left_on="district", right_on="자치구", how="left").drop(columns=["자치구"])
            d["network_decline_score"] = pct_rank_low(d["순증감"])
            d["priority_score"] = 0.6*d["nonfuel_demand_score"] + 0.4*d["network_decline_score"]
        else:
            d["priority_score"] = d["nonfuel_demand_score"]
        top = d.sort_values("priority_score", ascending=False).head(8)
        st.markdown("### 3. 유외수익/Network 방어 추가점검 지역")
        st.dataframe(
            top[["district","vehicles_202607","floating","resident","worker","apt_households","순증감","priority_score"]]
            .rename(columns={
                "district":"자치구","vehicles_202607":"자동차","floating":"길단위인구","resident":"상주인구",
                "worker":"직장인구","apt_households":"아파트 세대수","순증감":"Network 순증감","priority_score":"추가점검 점수"
            })
            .style.format({
                "자동차":"{:,.0f}","길단위인구":"{:,.0f}","상주인구":"{:,.0f}","직장인구":"{:,.0f}",
                "아파트 세대수":"{:,.0f}","Network 순증감":"{:+.0f}","추가점검 점수":"{:.0f}"
            }),
            hide_index=True, use_container_width=True
        )
        st.caption("추가점검 점수는 실제 수익성 점수가 아닙니다. 지역 수요와 과거 Network 감소를 이용해 현장 확인 순서를 정하는 스크리닝 지표입니다.")

    st.success(
        "프로그램의 최종 메시지: 최고가격제로 가격지원 여력이 제약될 수 있고 서울의 물리적 주유소 Network도 장기적으로 감소하는 만큼, "
        "DC·AC 경쟁력 강화는 가격만이 아니라 기존 주유소 자산의 수익성까지 함께 봐야 합니다. "
        "따라서 개별 GS 주유소에서는 세차장·경정비·편의점 등 이미 보유한 유외시설의 활용도를 가장 먼저 확인하고, "
        "그 다음 외부 파트너형 공간활용을 검토한 뒤 신규 CAPEX는 후순위로 둡니다."
    )
