"""
Vehicle Load Prediction Dashboard
Flipkart — Hajipur Mother Hub

Reads floor load data (Bag, Semi-Large, Tote, Secondary Pending) from Google Sheets,
aggregates by DH and cutoff, then predicts the optimal dispatch vehicle using a
RandomForest model trained on synthetic data generated from the business capacity rules.
"""

import re
import warnings
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_OK = True
except ImportError:
    GSPREAD_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── Constants ──────────────────────────────────────────────────────────────────

SPREADSHEET_ID = "1SbLc5pt0YPDBEQVOaOfyd-AJfvhTthQ5zUAcGgFU7Tc"

GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Business capacity rules per 32 Ft truck
BAG_SHIPMENTS_PER_32FT = 17_000   # max shipments (in bags) per 32ft truck
SEMI_PER_32FT          = 1_800    # max semi-large units per 32ft truck
TOTES_PER_32FT         = 650      # max totes per 32ft truck
SECONDARY_PER_32FT     = 14_235   # max regular shipments per 32ft truck
SHIPMENTS_PER_BAG      = 30       # average shipments per bag

# Fallback vehicle capacity table (overridden by Load Capacity sheet if present)
DEFAULT_VEHICLE_CAPS = [
    ("6.5 Ft",  800),
    ("8 Ft",   1_384),
    ("10 Ft",  2_051),
    ("14 Ft",  2_807),
    ("17 Ft",  3_359),
    ("20 Ft",  5_087),
    ("22 Ft",  5_865),
    ("24 Ft",  6_300),
    ("32 Ft", 14_235),
]

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🚛 Vehicle Load Predictor | Hajipur MH",
    layout="wide",
    page_icon="🚛",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background: #f8fafc; }
    section[data-testid="stSidebar"] { background: #1e293b !important; }
    section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid var(--accent);
        margin-bottom: 8px;
    }
    .metric-label { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
    .metric-value { font-size: 36px; font-weight: 800; color: var(--accent); line-height: 1.1; margin: 4px 0; }
    .metric-sub   { font-size: 12px; color: #94a3b8; }
    .predict-card {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        border-radius: 16px;
        padding: 26px 28px;
        color: white;
        box-shadow: 0 8px 28px rgba(37,99,235,.35);
    }
    .pred-label  { font-size: 13px; opacity: .8; font-weight: 500; }
    .pred-vehicle{ font-size: 44px; font-weight: 900; margin: 8px 0 4px; letter-spacing: -1px; }
    .pred-meta   { font-size: 13px; opacity: .75; margin-top: 10px; }
    .alt-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .bar-track {
        background: #f1f5f9;
        border-radius: 999px;
        height: 10px;
        margin-top: 4px;
        overflow: hidden;
    }
    .bar-fill { height: 10px; border-radius: 999px; }
    .step-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 4px;
    }
    div[data-testid="stSelectbox"] label { font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Auth / data loading ────────────────────────────────────────────────────────

def _gspread_client():
    if not GSPREAD_OK:
        st.error("Install gspread + google-auth: `pip install gspread google-auth`")
        st.stop()
    key_dict = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
    creds = Credentials.from_service_account_info(key_dict, scopes=GSHEETS_SCOPES)
    return gspread.authorize(creds)


@st.cache_data(ttl=300, show_spinner=False)
def load_all_sheets() -> dict:
    gc = _gspread_client()
    sh = gc.open_by_key(SPREADSHEET_ID)
    return {ws.title: ws.get_all_values() for ws in sh.worksheets()}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase, strip underscores/spaces/hyphens for fuzzy matching."""
    return re.sub(r"[_\s\-]+", "", str(s)).lower()


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _find_sheet(sheets: dict, *keywords):
    """Return (name, values) for first sheet whose name contains any keyword."""
    for kw in keywords:
        for name, vals in sheets.items():
            if kw.lower() in name.lower():
                return name, vals
    return None, []


def _to_df(values: list, header_row: int = 0) -> pd.DataFrame:
    if len(values) <= header_row:
        return pd.DataFrame()
    headers = [str(h).strip() for h in values[header_row]]
    rows = values[header_row + 1:]
    n = len(headers)
    rows = [r[:n] + [""] * max(0, n - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=headers)
    df.replace("", np.nan, inplace=True)
    return df.dropna(how="all")


def _dest_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "dest" in c.lower():
            return c
    return None


# ── Data parsing ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def parse_data(sheets_key: tuple):
    # re-load from cache — sheets_key is just used as a cache key
    sheets = load_all_sheets()

    # ── Bag ──────────────────────────────────────────────────────────────────
    _, bag_vals = _find_sheet(sheets, "bag")
    df_bag = pd.DataFrame()
    if bag_vals:
        df_bag = _to_df(bag_vals)
        dest_c = _dest_col(df_bag)
        # Tracking ID Count = shipments inside the bag
        tid_c = next(
            (c for c in df_bag.columns if "tracking" in c.lower() and "count" in c.lower()),
            None,
        )
        if dest_c and tid_c:
            df_bag = df_bag[[dest_c, tid_c]].rename(
                columns={dest_c: "destination", tid_c: "shipment_count"}
            )
            df_bag["destination"] = df_bag["destination"].astype(str).str.strip()
            df_bag["shipment_count"] = pd.to_numeric(
                df_bag["shipment_count"], errors="coerce"
            ).fillna(0).astype(int)
            df_bag = df_bag[df_bag["destination"].notna() & (df_bag["destination"] != "nan")]
        else:
            df_bag = pd.DataFrame()

    # ── Semi-large ────────────────────────────────────────────────────────────
    _, semi_vals = _find_sheet(sheets, "semi")
    df_semi = pd.DataFrame()
    if semi_vals:
        df_semi = _to_df(semi_vals)
        dest_c = _dest_col(df_semi)
        if dest_c:
            df_semi = df_semi[[dest_c]].rename(columns={dest_c: "destination"})
            df_semi["destination"] = df_semi["destination"].astype(str).str.strip()
            df_semi = df_semi[df_semi["destination"].notna() & (df_semi["destination"] != "nan")]

    # ── Tote ──────────────────────────────────────────────────────────────────
    _, tote_vals = _find_sheet(sheets, "tote")
    df_tote = pd.DataFrame()
    if tote_vals:
        df_tote = _to_df(tote_vals)
        dest_c = _dest_col(df_tote)
        if dest_c:
            df_tote = df_tote[[dest_c]].rename(columns={dest_c: "destination"})
            df_tote["destination"] = df_tote["destination"].astype(str).str.strip()
            df_tote = df_tote[df_tote["destination"].notna() & (df_tote["destination"] != "nan")]

    # ── Secondary pending ─────────────────────────────────────────────────────
    _, sec_vals = _find_sheet(sheets, "secondary")
    df_sec = pd.DataFrame()
    if sec_vals:
        df_sec = _to_df(sec_vals)
        nexthop_c = next(
            (c for c in df_sec.columns if "nexthop" in c.lower()),
            next((c for c in df_sec.columns if "next" in c.lower()), None),
        )
        if nexthop_c:
            df_sec = df_sec[[nexthop_c]].rename(columns={nexthop_c: "destination"})
            df_sec["destination"] = df_sec["destination"].astype(str).str.strip()
            df_sec = df_sec[df_sec["destination"].notna() & (df_sec["destination"] != "nan")]

    # ── Load Capacity ─────────────────────────────────────────────────────────
    _, cap_vals = _find_sheet(sheets, "load capacity", "capacity")
    vehicle_caps = list(DEFAULT_VEHICLE_CAPS)
    if cap_vals:
        parsed = []
        for row in cap_vals:
            if len(row) < 2:
                continue
            size_s = str(row[0]).strip()
            cnt_s  = str(row[1]).strip().replace(",", "")
            if re.search(r"\d+\s*(ft|feet)", size_s, re.I) and cnt_s.isdigit():
                parsed.append((size_s, int(cnt_s)))
        if parsed:
            vehicle_caps = parsed

    # ── DH Name Cut-Off Wise ──────────────────────────────────────────────────
    _, dh_vals = _find_sheet(sheets, "dh name", "cut-off", "cutoff", "dh")
    df_dh = pd.DataFrame()
    if dh_vals:
        # Detect header row (look for a row mentioning "dh" or "cutoff")
        hdr_idx = 0
        for i, row in enumerate(dh_vals[:6]):
            joined = " ".join(str(c).lower() for c in row)
            if any(kw in joined for kw in ("dh", "cutoff", "cut", "shift")):
                hdr_idx = i
                break

        df_dh = _to_df(dh_vals, hdr_idx)
        cols = list(df_dh.columns)
        if len(cols) >= 4:
            # Auto-detect cutoff column (contains HH:MM or HH:MM:SS patterns)
            cutoff_col = None
            for c in cols:
                sample = df_dh[c].dropna().astype(str).head(30)
                if sample.str.match(r"^\d{1,2}:\d{2}").sum() >= 3:
                    cutoff_col = c
                    break

            dh_code_col = cols[0]
            dh_name_col = cols[1]

            if cutoff_col:
                keep = [dh_code_col, dh_name_col]
                # Also grab nexthop (col 2) if it exists and cutoff is col 3+
                if len(cols) > 2 and cols[2] != cutoff_col:
                    keep.append(cols[2])
                keep.append(cutoff_col)

                df_dh = df_dh[keep].copy()
                df_dh.columns = (
                    ["dh_code", "dh_name", "nexthop", "cutoff"]
                    if len(keep) == 4
                    else ["dh_code", "dh_name", "cutoff"]
                )
                df_dh = df_dh[df_dh["dh_name"].astype(str).str.strip() != ""]
                df_dh = df_dh[df_dh["dh_name"].astype(str) != "nan"]
                df_dh["cutoff"] = df_dh["cutoff"].astype(str).str.strip()
                # Keep only rows with valid time
                df_dh = df_dh[df_dh["cutoff"].str.match(r"^\d{1,2}:\d{2}", na=False)]
                # Normalise to HH:MM for display
                df_dh["cutoff_display"] = df_dh["cutoff"].str[:5]

    return df_bag, df_semi, df_tote, df_sec, vehicle_caps, df_dh


# ── Load matching ──────────────────────────────────────────────────────────────

def _match_dest(df: pd.DataFrame, dh_name: str, nexthop: str = "") -> pd.DataFrame:
    """Return rows whose destination matches dh_name or nexthop (case-insensitive → fuzzy)."""
    if df.empty:
        return df
    targets = {_norm(dh_name)}
    if nexthop and nexthop.lower() not in ("direct", "null", "nan", ""):
        targets.add(_norm(nexthop))

    norms = df["destination"].apply(_norm)
    # Exact norm match
    exact = df[norms.isin(targets)]
    if not exact.empty:
        return exact
    # Fuzzy fallback (threshold 0.72)
    scores = norms.apply(lambda n: max(_fuzzy(n, t) for t in targets))
    return df[scores >= 0.72]


def get_dh_load(dh_name: str, nexthop: str, df_bag, df_semi, df_tote, df_sec) -> dict:
    bag_rows  = _match_dest(df_bag,  dh_name, nexthop)
    semi_rows = _match_dest(df_semi, dh_name, nexthop)
    tote_rows = _match_dest(df_tote, dh_name, nexthop)
    sec_rows  = _match_dest(df_sec,  dh_name, nexthop)

    bag_count     = len(bag_rows)
    bag_shipments = int(bag_rows["shipment_count"].sum()) if not bag_rows.empty else 0
    semi_count    = len(semi_rows)
    tote_count    = len(tote_rows)
    sec_count     = len(sec_rows)

    return dict(
        bag_count=bag_count,
        bag_shipments=bag_shipments,
        semi_count=semi_count,
        tote_count=tote_count,
        secondary_count=sec_count,
    )


# ── ML model ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_model(vehicle_caps_key: tuple):
    """
    Train a RandomForest on synthetic data derived from business capacity rules.
    Features: [bag_shipments, semi_count, tote_count, secondary_count]
    Label   : vehicle size
    """
    vehicle_caps = list(vehicle_caps_key)
    max_cap      = max(c for _, c in vehicle_caps)
    vehicles     = [v for v, _ in vehicle_caps]
    caps         = [c for _, c in vehicle_caps]

    rng = np.random.default_rng(42)
    n   = 30_000
    bag_s  = rng.integers(0, BAG_SHIPMENTS_PER_32FT * 2 + 1, n)
    semi_s = rng.integers(0, SEMI_PER_32FT * 2 + 1, n)
    tote_s = rng.integers(0, TOTES_PER_32FT * 2 + 1, n)
    sec_s  = rng.integers(0, SECONDARY_PER_32FT * 2 + 1, n)

    fracs  = (
        bag_s  / BAG_SHIPMENTS_PER_32FT
        + semi_s / SEMI_PER_32FT
        + tote_s / TOTES_PER_32FT
        + sec_s  / SECONDARY_PER_32FT
    )
    required = fracs * max_cap

    labels = []
    for req in required:
        chosen = vehicles[-1]
        for v, c in zip(vehicles, caps):
            if c >= req:
                chosen = v
                break
        labels.append(chosen)

    X = np.column_stack([bag_s, semi_s, tote_s, sec_s])
    le = LabelEncoder()
    le.fit(vehicles)
    y = le.transform(labels)

    clf = RandomForestClassifier(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf, le


def predict_vehicle(load: dict, vehicle_caps: list):
    """
    Returns (recommended_vehicles, confidence_pct, alternatives, utilisation_frac).
    Uses RandomForest when sklearn is available, falls back to rule-based.
    """
    max_cap = max(c for _, c in vehicle_caps)
    vehicles_list = [v for v, _ in vehicle_caps]
    caps_list     = [c for _, c in vehicle_caps]

    bag_s  = load["bag_shipments"]
    semi_s = load["semi_count"]
    tote_s = load["tote_count"]
    sec_s  = load["secondary_count"]

    bag_frac  = bag_s  / BAG_SHIPMENTS_PER_32FT
    semi_frac = semi_s / SEMI_PER_32FT
    tote_frac = tote_s / TOTES_PER_32FT
    sec_frac  = sec_s  / SECONDARY_PER_32FT
    total_frac = bag_frac + semi_frac + tote_frac + sec_frac

    if total_frac == 0:
        return [], 0, [], 0.0

    # Multi-vehicle scenario
    n_trucks = int(np.ceil(total_frac))
    if n_trucks > 1:
        remainder_frac = total_frac - (n_trucks - 1)
        req_last = remainder_frac * max_cap
        last_v = next((v for v, c in vehicle_caps if c >= req_last), "32 Ft")
        chosen = ["32 Ft"] * (n_trucks - 1) + [last_v]
        return chosen, 88, [], total_frac

    # Single vehicle: use RandomForest if available
    if SKLEARN_OK:
        caps_key = tuple(vehicle_caps)
        clf, le = build_model(caps_key)
        X_pred  = np.array([[bag_s, semi_s, tote_s, sec_s]])
        proba   = clf.predict_proba(X_pred)[0]
        # Sort by descending probability
        order   = np.argsort(proba)[::-1]
        best_idx     = order[0]
        best_vehicle = le.classes_[best_idx]
        confidence   = round(float(proba[best_idx]) * 100, 1)

        # Utilisation inside the predicted vehicle
        pred_cap = next((c for v, c in vehicle_caps if v == best_vehicle), max_cap)
        util     = min(total_frac * max_cap / pred_cap, 1.0)

        # Alternatives (next 2 by probability)
        alts = []
        for idx in order[1:3]:
            alt_v   = le.classes_[idx]
            alt_cap = next((c for v, c in vehicle_caps if v == alt_v), max_cap)
            alt_util = min(total_frac * max_cap / alt_cap, 1.0)
            alts.append((alt_v, round(float(proba[idx]) * 100, 1), round(alt_util * 100)))

        return [best_vehicle], round(confidence), alts, round(util, 3)

    # Fallback: rule-based
    required = total_frac * max_cap
    for v, c in vehicle_caps:
        if c >= required:
            util = required / c
            conf = 90 if 0.6 <= util <= 0.85 else (70 if util < 0.6 else 75)
            # next 2 alternatives
            remaining = [(av, ac) for av, ac in vehicle_caps if ac > c][:2]
            alts = [(av, 0, round(required / ac * 100)) for av, ac in remaining]
            return [v], conf, alts, round(util, 3)

    return ["32 Ft"], 60, [], 1.0


# ── UI helpers ─────────────────────────────────────────────────────────────────

def metric_card(label: str, value, sub: str = "", accent: str = "#2563eb"):
    st.markdown(
        f"""<div class="metric-card" style="--accent:{accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def progress_bar(label: str, val: int, cap: int, unit: str, color: str):
    pct = min(100, round(val / cap * 100, 1)) if cap else 0
    st.markdown(
        f"""<div style="margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
                <span style="font-weight:600;color:#334155">{label}</span>
                <span style="color:#64748b">{val:,} / {cap:,} {unit} &nbsp;·&nbsp; <b>{pct}%</b></span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct}%;background:{color}"></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🚛 Vehicle Predictor")
        st.caption("Flipkart · Hajipur Mother Hub")
        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            """
- Reads live floor data from Google Sheets
- Aggregates Bags, Semi-Large, Totes & Secondary pending by DH
- A **RandomForest ML model** (trained on 30 K synthetic samples derived from capacity rules) predicts the optimal vehicle
- Confidence = model probability score
""",
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown(
            f"<div style='font-size:11px;color:#94a3b8;margin-top:8px'>Cache TTL: 5 min</div>",
            unsafe_allow_html=True,
        )

    # ── Title ──────────────────────────────────────────────────────────────────
    st.markdown("## 🚛 Vehicle Load Prediction Dashboard")
    st.markdown("*Floor load analysis + ML-based vehicle recommendation · Hajipur MH*")
    st.divider()

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data from Google Sheets…"):
        try:
            raw = load_all_sheets()
            sheet_key = tuple(sorted(raw.keys()))
            df_bag, df_semi, df_tote, df_sec, vehicle_caps, df_dh = parse_data(sheet_key)
        except Exception as exc:
            st.error(f"❌ Could not load sheet: {exc}")
            st.info("Make sure `GOOGLE_SERVICE_ACCOUNT` is set in Streamlit secrets and the sheet is shared with the service account.")
            st.stop()

    # Dataset summary
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Bags on Floor",      f"{len(df_bag):,}",  f"{df_bag['shipment_count'].sum() if not df_bag.empty else 0:,} shipments", "#f59e0b")
    with c2:
        metric_card("Semi-Large Shipments",     f"{len(df_semi):,}", "Floor pending", "#3b82f6")
    with c3:
        metric_card("Totes on Floor",           f"{len(df_tote):,}", "Pending dispatch", "#8b5cf6")
    with c4:
        metric_card("Secondary Pending",        f"{len(df_sec):,}",  "Sorted, not bagged", "#ef4444")

    st.divider()

    # ── Step 1: Cutoff ─────────────────────────────────────────────────────────
    if df_dh.empty:
        st.warning("⚠️ DH Name Cut-Off data not found. Check sheet name matches 'DH Name Cut-Off Wise'.")
        return

    cutoffs = sorted(df_dh["cutoff_display"].dropna().unique().tolist())
    if not cutoffs:
        st.warning("No cutoff times parsed.")
        return

    st.markdown('<div class="step-header">Step 1 — Choose Cutoff</div>', unsafe_allow_html=True)
    selected_cutoff = st.selectbox(
        "🕐 TMS Cutoff", cutoffs,
        help="All DHs dispatching at this cut-off time will be listed in Step 2"
    )

    filtered_dh = df_dh[df_dh["cutoff_display"] == selected_cutoff].copy()
    dh_options  = sorted(filtered_dh["dh_name"].dropna().unique().tolist())

    if not dh_options:
        st.info("No DHs found for this cutoff.")
        return

    # ── Step 2: DH ─────────────────────────────────────────────────────────────
    st.markdown(f'<div class="step-header" style="margin-top:16px">Step 2 — Choose Destination Hub &nbsp;·&nbsp; {len(dh_options)} DHs at {selected_cutoff}</div>', unsafe_allow_html=True)
    selected_dh = st.selectbox("🏭 DH Name", dh_options)

    # Get nexthop for better matching
    row_dh  = filtered_dh[filtered_dh["dh_name"] == selected_dh].iloc[0]
    nexthop = str(row_dh.get("nexthop", "")).strip() if "nexthop" in row_dh.index else ""
    dh_code = str(row_dh.get("dh_code", "")).strip()

    # ── Load Analysis ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 📦 Floor Load — `{selected_dh}` &nbsp; `{dh_code}` &nbsp; ⏱ cutoff {selected_cutoff}")

    with st.spinner("Aggregating floor data…"):
        load = get_dh_load(selected_dh, nexthop, df_bag, df_semi, df_tote, df_sec)

    total_shipments = (
        load["bag_shipments"]
        + load["semi_count"]
        + load["tote_count"]
        + load["secondary_count"]
    )

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("🛍️ Bags",         f"{load['bag_count']:,}",  f"{load['bag_shipments']:,} shipments inside", "#f59e0b")
    with m2:
        metric_card("📦 Semi-Large",   f"{load['semi_count']:,}", "Shipments on floor", "#3b82f6")
    with m3:
        metric_card("🧺 Totes",        f"{load['tote_count']:,}", "Totes on floor", "#8b5cf6")
    with m4:
        metric_card("📋 Secondary",    f"{load['secondary_count']:,}", "Sorted, pending bag", "#ef4444")

    if total_shipments == 0:
        st.success(f"✅ No pending floor load for **{selected_dh}**. All clear!")
        return

    st.markdown(f"<br><div style='font-size:15px;color:#475569'>**Total equivalent shipments:** <b style='font-size:22px;color:#1e293b'>{total_shipments:,}</b></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction + Pie ───────────────────────────────────────────────────────
    col_pie, col_pred = st.columns([1, 1])

    with col_pie:
        labels = ["Bag Shipments", "Semi-Large", "Totes", "Secondary Pending"]
        values = [
            load["bag_shipments"], load["semi_count"],
            load["tote_count"],    load["secondary_count"],
        ]
        colors = ["#f59e0b", "#3b82f6", "#8b5cf6", "#ef4444"]
        non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]

        fig = go.Figure(go.Pie(
            labels=[x[0] for x in non_zero],
            values=[x[1] for x in non_zero],
            marker_colors=[x[2] for x in non_zero],
            hole=0.55,
            textinfo="label+percent",
            textfont_size=12,
        ))
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            annotations=[dict(
                text=f"<b>{total_shipments:,}</b><br><span style='font-size:11px'>Total</span>",
                x=0.5, y=0.5, font_size=15, showarrow=False,
            )],
        )
        st.markdown("**Load Bifurcation**")
        st.plotly_chart(fig, use_container_width=True)

    with col_pred:
        vehicles, confidence, alternatives, util = predict_vehicle(load, vehicle_caps)

        if not vehicles:
            st.warning("Could not predict vehicle — check load data.")
        else:
            veh_str  = " + ".join(vehicles)
            conf_color = (
                "#16a34a" if confidence >= 75
                else "#f59e0b" if confidence >= 55
                else "#ef4444"
            )
            util_pct = round(util * 100) if util <= 1 else round(util * 100)

            st.markdown(
                f"""<div class="predict-card">
                    <div class="pred-label">🤖 ML Recommendation (RandomForest)</div>
                    <div class="pred-vehicle">{veh_str}</div>
                    <div style="display:flex;gap:24px;margin-top:12px">
                        <div>
                            <div style="font-size:12px;opacity:.75">Confidence</div>
                            <div style="font-size:26px;font-weight:800;color:{conf_color}">{confidence}%</div>
                        </div>
                        <div>
                            <div style="font-size:12px;opacity:.75">Load Utilization</div>
                            <div style="font-size:26px;font-weight:800">{min(util_pct,100)}%</div>
                        </div>
                        <div>
                            <div style="font-size:12px;opacity:.75">Trucks Needed</div>
                            <div style="font-size:26px;font-weight:800">{len(vehicles)}</div>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            if alternatives:
                st.markdown("<br><b>Alternative Vehicles</b>", unsafe_allow_html=True)
                for alt_v, alt_conf, alt_util_pct in alternatives:
                    st.markdown(
                        f"""<div class="alt-card">
                            <span style="font-weight:700;color:#1e293b">{alt_v}</span>
                            <span style="color:#64748b;font-size:13px">
                                {alt_util_pct}% load &nbsp;·&nbsp; {alt_conf}% confidence
                            </span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # ── Capacity bars ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Capacity Utilization *(per type, relative to 32 Ft truck)*")

    bars = [
        ("🛍️ Bags",           load["bag_count"],      BAG_SHIPMENTS_PER_32FT // SHIPMENTS_PER_BAG, "bags",      "#f59e0b"),
        ("📦 Semi-Large",     load["semi_count"],     SEMI_PER_32FT,                               "shipments", "#3b82f6"),
        ("🧺 Totes",          load["tote_count"],     TOTES_PER_32FT,                              "totes",     "#8b5cf6"),
        ("📋 Secondary",      load["secondary_count"],SECONDARY_PER_32FT,                          "shipments", "#ef4444"),
    ]
    for label, val, cap, unit, color in bars:
        progress_bar(label, val, cap, unit, color)

    # ── Gauge ──────────────────────────────────────────────────────────────────
    max_cap = max(c for _, c in vehicle_caps)
    total_frac = (
        load["bag_shipments"] / BAG_SHIPMENTS_PER_32FT
        + load["semi_count"]  / SEMI_PER_32FT
        + load["tote_count"]  / TOTES_PER_32FT
        + load["secondary_count"] / SECONDARY_PER_32FT
    )
    overall_pct = round(min(total_frac, 1.0) * 100, 1)

    _, gcol, _ = st.columns([1, 1.4, 1])
    with gcol:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_pct,
            number={"suffix": "%", "font": {"size": 38, "color": "#1e293b"}},
            title={"text": "Overall 32 Ft Equivalent Utilization", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#2563eb", "thickness": 0.28},
                "steps": [
                    {"range": [0, 50],  "color": "#dbeafe"},
                    {"range": [50, 80], "color": "#bfdbfe"},
                    {"range": [80, 100],"color": "#fef3c7"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.75, "value": 90,
                },
            },
        ))
        fig_g.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig_g, use_container_width=True)

    if total_frac > 1:
        trucks = int(np.ceil(total_frac))
        st.warning(
            f"⚠️ Load exceeds 1 truck capacity by **{round((total_frac-1)*100)}%** "
            f"— minimum **{trucks} vehicles** required."
        )

    # ── Vehicle Reference ──────────────────────────────────────────────────────
    with st.expander("🚛 Vehicle Capacity Reference Table", expanded=False):
        max_v = max(c for _, c in vehicle_caps)
        cap_df = pd.DataFrame(vehicle_caps, columns=["Vehicle", "Max Shipments (32 Ft equiv.)"])
        cap_df["% of 32 Ft"] = cap_df["Max Shipments (32 Ft equiv.)"].apply(
            lambda x: f"{x/max_v*100:.1f}%"
        )
        cap_df["Bags Capacity"]      = (cap_df["Max Shipments (32 Ft equiv.)"] / max_v * (BAG_SHIPMENTS_PER_32FT // SHIPMENTS_PER_BAG)).astype(int)
        cap_df["Semi-Large Capacity"]= (cap_df["Max Shipments (32 Ft equiv.)"] / max_v * SEMI_PER_32FT).astype(int)
        cap_df["Totes Capacity"]     = (cap_df["Max Shipments (32 Ft equiv.)"] / max_v * TOTES_PER_32FT).astype(int)
        st.dataframe(cap_df, use_container_width=True, hide_index=True)

    # ── All DHs for this cutoff ────────────────────────────────────────────────
    with st.expander(f"📋 All DHs at cutoff {selected_cutoff} ({len(dh_options)} total)", expanded=False):
        st.dataframe(
            filtered_dh[["dh_code", "dh_name"]].reset_index(drop=True).rename(
                columns={"dh_code": "DH Code", "dh_name": "DH Name"}
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
