"""
Vehicle Load Prediction Dashboard  —  Flipkart · Hajipur Mother Hub
• Multi-select cutoff + DH tables
• Smart vehicle recommendation targeting 100 % utilisation
• Manual vehicle selector with remaining-capacity breakdown
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

SPREADSHEET_ID = "1SbLc5pt0YPDBEQVOaOfyd-AJfvhTthQ5zUAcGgFU7Tc"
GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

BAG_SHIPMENTS_32FT = 17_000   
SEMI_32FT          = 1_800    
TOTES_32FT         = 650      
SECONDARY_32FT     = 14_235   
SHIPMENTS_PER_BAG  = 30

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

TARGET_UTIL = 1.00   

st.set_page_config(
    page_title="🚛 Vehicle Load Predictor | Hajipur MH",
    layout="wide",
    page_icon="🚛",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp{background:#f0f2f6}
section[data-testid="stSidebar"]{background:#1e293b!important}
section[data-testid="stSidebar"] *{color:#f1f5f9!important}
.kcard{background:var(--ac);border-radius:14px;padding:16px 20px;
       box-shadow:0 4px 14px rgba(0,0,0,.15)}
/* Base .klabel/.kvalue/.ksub are reused on white-background detail boxes
   elsewhere, so they stay dark by default; .kcard scopes its own white text. */
.klabel{font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.6px}
.kvalue{font-size:34px;font-weight:900;color:var(--ac,#2563eb);line-height:1.1;margin:2px 0}
.ksub  {font-size:12px;color:#94a3b8;margin-top:2px}
.kcard .klabel{color:rgba(255,255,255,.85)}
.kcard .kvalue{color:#ffffff}
.kcard .ksub  {color:rgba(255,255,255,.75)}
.predcard{background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:16px;
          padding:24px 26px;color:white;box-shadow:0 8px 28px rgba(37,99,235,.35)}
.sec-hdr{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
         color:#64748b;margin:0 0 6px}
.bartrack{background:#e2e8f0;border-radius:999px;height:11px;overflow:hidden;margin-top:3px}
.barfill{height:11px;border-radius:999px}
.vcap-row{background:white;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;
          margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}

/* Dashboard name pinned in the sticky app header — always visible, never scrolls away */
header[data-testid="stHeader"]{height:56px}
header[data-testid="stHeader"]::before{
    content:"🚛  Vehicle Load Prediction Dashboard";
    position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
    font-size:24px; font-weight:800; color:#1e293b; white-space:nowrap;
    pointer-events:none;
}
</style>
""", unsafe_allow_html=True)

def _gc():
    if not GSPREAD_OK:
        st.error("Install gspread + google-auth")
        st.stop()
    key = "gcp_service_account" if "gcp_service_account" in st.secrets else "GOOGLE_SERVICE_ACCOUNT"
    creds = Credentials.from_service_account_info(dict(st.secrets[key]), scopes=GSHEETS_SCOPES)
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner=False)
def load_sheets():
    gc = _gc()
    sh = gc.open_by_key(SPREADSHEET_ID)
    return {ws.title: ws.get_all_values() for ws in sh.worksheets()}

def _norm(s):
    return re.sub(r"[_\s\-]+", "", str(s)).lower()

def _fuzzy(a, b):
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _find(sheets, *kws):
    for kw in kws:
        for name, vals in sheets.items():
            if kw.lower() in name.lower():
                return vals
    return []

def _df(vals, hdr=0):
    if len(vals) <= hdr:
        return pd.DataFrame()
    heads = [str(h).strip() for h in vals[hdr]]
    rows  = vals[hdr+1:]
    n = len(heads)
    rows = [r[:n] + [""] * max(0, n-len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=heads)
    df.replace("", np.nan, inplace=True)
    return df.dropna(how="all")

def _destcol(df):
    return next((c for c in df.columns if "dest" in c.lower()), None)

@st.cache_data(ttl=300, show_spinner=False)
def parse(_key):
    sheets = load_sheets()

    # Bag
    bag_v = _find(sheets, "bag")
    df_bag = pd.DataFrame()
    if bag_v:
        d = _df(bag_v)
        dc = _destcol(d)
        tc = next((c for c in d.columns if "tracking" in c.lower() and "count" in c.lower()), None)
        if dc and tc:
            df_bag = d[[dc, tc]].rename(columns={dc:"destination", tc:"ship_count"})
            df_bag["destination"] = df_bag["destination"].astype(str).str.strip()
            df_bag["ship_count"]  = pd.to_numeric(df_bag["ship_count"], errors="coerce").fillna(0).astype(int)
            df_bag = df_bag[df_bag["destination"].notna() & (df_bag["destination"] != "nan")]

    semi_v = _find(sheets, "semi")
    df_semi = pd.DataFrame()
    if semi_v:
        d = _df(semi_v); dc = _destcol(d)
        if dc:
            df_semi = d[[dc]].rename(columns={dc:"destination"})
            df_semi["destination"] = df_semi["destination"].astype(str).str.strip()
            df_semi = df_semi[df_semi["destination"].notna() & (df_semi["destination"] != "nan")]

    tote_v = _find(sheets, "tote")
    df_tote = pd.DataFrame()
    if tote_v:
        d = _df(tote_v); dc = _destcol(d)
        if dc:
            df_tote = d[[dc]].rename(columns={dc:"destination"})
            df_tote["destination"] = df_tote["destination"].astype(str).str.strip()
            df_tote = df_tote[df_tote["destination"].notna() & (df_tote["destination"] != "nan")]

    sec_v = _find(sheets, "secondary")
    df_sec = pd.DataFrame()
    if sec_v:
        d = _df(sec_v)
        nc = next((c for c in d.columns if "nexthop" in c.lower()), None)
        if nc:
            df_sec = d[[nc]].rename(columns={nc:"destination"})
            df_sec["destination"] = df_sec["destination"].astype(str).str.strip()
            df_sec = df_sec[df_sec["destination"].notna() & (df_sec["destination"] != "nan")]

    # Bagging Pending — same treatment as Secondary Pending: each row is one
    # shipment waiting to be bagged, folded into bag counts at 30/bag. Its
    # destination column is "shipment_facility_name" (not the *_id column).
    bagging_v = _find(sheets, "bagging")
    if bagging_v:
        d = _df(bagging_v)
        nc = next(
            (c for c in d.columns if "facility" in c.lower() and "name" in c.lower() and "id" not in c.lower()),
            None,
        )
        if nc:
            df_bagging = d[[nc]].rename(columns={nc:"destination"})
            df_bagging["destination"] = df_bagging["destination"].astype(str).str.strip()
            df_bagging = df_bagging[df_bagging["destination"].notna() & (df_bagging["destination"] != "nan")]
            df_sec = pd.concat([df_sec, df_bagging], ignore_index=True) if not df_sec.empty else df_bagging

    cap_v = _find(sheets, "load capacity", "capacity")
    vcaps = list(DEFAULT_VEHICLE_CAPS)
    if cap_v:
        parsed = []
        for row in cap_v:
            if len(row) < 2:
                continue
            s = str(row[0]).strip(); n = str(row[1]).strip().replace(",","")
            if re.search(r"\d+\s*(ft|feet)", s, re.I) and n.isdigit():
                parsed.append((s, int(n)))
        if parsed:
            vcaps = parsed

    dh_v = _find(sheets, "dh name", "cut-off", "cutoff", "dh")
    df_dh = pd.DataFrame()
    if dh_v:
        hi = 0
        for i, row in enumerate(dh_v[:6]):
            j = " ".join(str(c).lower() for c in row)
            if any(k in j for k in ("dh","cutoff","cut","shift")):
                hi = i; break
        d = _df(dh_v, hi)
        cols = list(d.columns)
        if len(cols) >= 3:
            coc = next(
                (c for c in cols if d[c].dropna().astype(str).head(30).str.match(r"^\d{1,2}:\d{2}").sum() >= 3),
                None,
            )
            if coc:
                keep = [cols[0], cols[1]]
                if len(cols) > 2 and cols[2] != coc:
                    keep.append(cols[2])
                keep.append(coc)
                d = d[keep].copy()
                d.columns = (["dh_code","dh_name","nexthop","cutoff"] if len(keep)==4
                             else ["dh_code","dh_name","cutoff"])
                d = d[d["dh_name"].astype(str).str.strip().ne("") & d["dh_name"].astype(str).ne("nan")]
                d["cutoff"] = d["cutoff"].astype(str).str.strip()
                d = d[d["cutoff"].str.match(r"^\d{1,2}:\d{2}", na=False)]
                d["cutoff_display"] = d["cutoff"].str[:5]
                df_dh = d

    return df_bag, df_semi, df_tote, df_sec, vcaps, df_dh


def _match(df, dh_name, nexthop=""):
    """Slow path (row-by-row fuzzy scan) — kept for single-DH lookups only."""
    if df.empty:
        return df
    targets = {_norm(dh_name)}
    if nexthop and nexthop.lower() not in ("direct","null","nan",""):
        targets.add(_norm(nexthop))
    norms = df["destination"].apply(_norm)
    exact = df[norms.isin(targets)]
    if not exact.empty:
        return exact
    scores = norms.apply(lambda n: max(_fuzzy(n, t) for t in targets))
    return df[scores >= 0.72]

def dh_load(dh_name, nexthop, df_bag, df_semi, df_tote, df_sec):
    br  = _match(df_bag,  dh_name, nexthop)
    sr  = _match(df_semi, dh_name, nexthop)
    tr  = _match(df_tote, dh_name, nexthop)
    secr= _match(df_sec,  dh_name, nexthop)
    return dict(
        bag_count     = len(br),
        bag_shipments = int(br["ship_count"].sum()) if not br.empty else 0,
        semi_count    = len(sr),
        tote_count    = len(tr),
        secondary_count = len(secr),
    )

def _agg_by_dest(df, value_col=None):
    """Aggregate a destination-keyed df ONCE into {norm_dest: (row_count, value_sum)}."""
    if df.empty:
        return {}, []
    norms = df["destination"].apply(_norm)
    if value_col:
        g = df.assign(_n=norms).groupby("_n")[value_col].agg(["size", "sum"])
        agg = {n: (int(r["size"]), int(r["sum"])) for n, r in g.iterrows()}
    else:
        g = norms.value_counts()
        agg = {n: (int(c), 0) for n, c in g.items()}
    return agg, list(agg.keys())

def _match_agg(dh_name, nexthop, agg, unique_norms):
    """O(unique destinations) lookup instead of O(rows) — exact first, fuzzy fallback."""
    if not agg:
        return 0, 0
    targets = {_norm(dh_name)}
    if nexthop and nexthop.lower() not in ("direct", "null", "nan", ""):
        targets.add(_norm(nexthop))

    exact_hits = [t for t in targets if t in agg]
    if exact_hits:
        cnt = sum(agg[t][0] for t in exact_hits)
        val = sum(agg[t][1] for t in exact_hits)
        return cnt, val

    cnt = val = 0
    for n in unique_norms:
        if max(_fuzzy(n, t) for t in targets) >= 0.72:
            c, v = agg[n]
            cnt += c
            val += v
    return cnt, val

@st.cache_data(ttl=300, show_spinner=False)
def compute_all_dh_loads(df_bag, df_semi, df_tote, df_sec, df_dh):
    """Pre-compute loads for every DH once and cache for 5 min.
    Aggregates each source sheet by unique destination ONCE, then does
    O(unique destinations) lookups per DH instead of O(rows) — this is what
    makes 1000+ DHs load instantly instead of taking minutes."""
    bag_agg, bag_norms   = _agg_by_dest(df_bag,  "ship_count")
    semi_agg, semi_norms = _agg_by_dest(df_semi)
    tote_agg, tote_norms = _agg_by_dest(df_tote)
    sec_agg,  sec_norms  = _agg_by_dest(df_sec)

    result = {}
    for _, dr in df_dh.drop_duplicates("dh_name").iterrows():
        dh_n = str(dr["dh_name"])
        nx   = str(dr.get("nexthop", "")) if "nexthop" in dr.index else ""

        bag_count, bag_ships = _match_agg(dh_n, nx, bag_agg,  bag_norms)
        semi_count, _        = _match_agg(dh_n, nx, semi_agg, semi_norms)
        tote_count, _        = _match_agg(dh_n, nx, tote_agg, tote_norms)
        sec_count, _         = _match_agg(dh_n, nx, sec_agg,  sec_norms)

        result[dh_n] = dict(
            bag_count       = bag_count,
            bag_shipments   = bag_ships,
            semi_count      = semi_count,
            tote_count      = tote_count,
            secondary_count = sec_count,
        )
    return result


def load_to_frac(load):
    return (
        load["bag_shipments"] / BAG_SHIPMENTS_32FT
        + load["semi_count"]  / SEMI_32FT
        + load["tote_count"]  / TOTES_32FT
        + load["secondary_count"] / SECONDARY_32FT
    )

def frac_to_equiv(frac, max_cap):
    """Fraction of 32Ft → equivalent shipment count."""
    return frac * max_cap

def recommend_vehicle(total_frac, vcaps):
    """
    Pick the smallest vehicle that fits the load (= highest utilisation).
    Returns (vehicle_name, capacity, utilisation_frac, trucks_needed, truck_breakdown).
    truck_breakdown is a list of {"vehicle", "capacity", "util_frac"} — one entry
    per truck — so multi-truck loads can show utilisation separately per vehicle
    instead of one blended number.
    """
    max_cap  = max(c for _, c in vcaps)
    req_cap  = total_frac * max_cap

    if total_frac == 0:
        return None, 0, 0.0, 0, []

    if total_frac > 1:
        n = int(np.ceil(total_frac))
        max_v = next(v for v, c in vcaps if c == max_cap)
        # last partial truck — utilisation must be relative to ITS OWN capacity,
        # not the max-cap capacity used to size the full trucks before it.
        rem_frac       = total_frac - (n - 1)
        rem_cap        = rem_frac * max_cap
        last_v, last_c = next(((v, c) for v, c in vcaps if c >= rem_cap), vcaps[-1])
        last_util      = rem_cap / last_c if last_c else 0.0
        breakdown = [{"vehicle": max_v, "capacity": max_cap, "util_frac": 1.0} for _ in range(n - 1)]
        breakdown.append({"vehicle": last_v, "capacity": last_c, "util_frac": last_util})
        # Collapse the label when the last (partial) truck is the SAME type
        # as the full trucks — e.g. "32 Ft × 2" instead of "32 Ft × 1 + 32 Ft".
        label = f"{max_v} × {n}" if last_v == max_v else f"{max_v} × {n-1} + {last_v}"
        return label, max_cap, last_util, n, breakdown

    for v, c in vcaps:
        if c >= req_cap:
            util = req_cap / c if c else 0.0
            return v, c, util, 1, [{"vehicle": v, "capacity": c, "util_frac": util}]

    v, c = vcaps[-1]
    util = req_cap / c if c else 0.0
    return v, c, util, 1, [{"vehicle": v, "capacity": c, "util_frac": util}]

def remaining_to_target(current_equiv, vehicle_cap, max_cap, target=TARGET_UTIL):
    """How many more equivalent shipments to reach target utilisation."""
    target_equiv = vehicle_cap * target
    return max(0, target_equiv - current_equiv)

def breakdown_remaining(equiv_remaining, max_cap):
    """Express remaining equivalent capacity as bags / semi-large / totes."""
    frac = equiv_remaining / max_cap if max_cap else 0
    return dict(
        bags      = int(frac * BAG_SHIPMENTS_32FT / SHIPMENTS_PER_BAG),
        semi      = int(frac * SEMI_32FT),
        totes     = int(frac * TOTES_32FT),
        secondary = int(frac * SECONDARY_32FT),
    )


# ── Table cell styling (pandas Styler — no extra deps needed) ────────────────
def _status_dot(util_pct):
    """🟢/🟡/🔴 status dot for a utilization percentage."""
    if util_pct >= 75:
        return "🟢"
    if util_pct >= 40:
        return "🟡"
    return "🔴"

def _util_badge_style(v):
    if v >= 75:
        return "background-color:#dcfce7;color:#166534;font-weight:700;border-radius:6px"
    if v >= 40:
        return "background-color:#fef9c3;color:#854d0e;font-weight:700;border-radius:6px"
    return "background-color:#fee2e2;color:#991b1b;font-weight:700;border-radius:6px"

_VEHICLE_BADGE_TIERS = {
    "6.5 Ft": "background-color:#dbeafe;color:#1e40af;font-weight:700;border-radius:6px",
    "8 Ft":   "background-color:#dbeafe;color:#1e40af;font-weight:700;border-radius:6px",
    "10 Ft":  "background-color:#dbeafe;color:#1e40af;font-weight:700;border-radius:6px",
    "14 Ft":  "background-color:#e0e7ff;color:#4338ca;font-weight:700;border-radius:6px",
    "17 Ft":  "background-color:#e0e7ff;color:#4338ca;font-weight:700;border-radius:6px",
    "20 Ft":  "background-color:#fef3c7;color:#92400e;font-weight:700;border-radius:6px",
    "22 Ft":  "background-color:#fef3c7;color:#92400e;font-weight:700;border-radius:6px",
    "24 Ft":  "background-color:#fef3c7;color:#92400e;font-weight:700;border-radius:6px",
    "32 Ft":  "background-color:#fae8ff;color:#86198f;font-weight:700;border-radius:6px",
}

def _vehicle_badge_style(v):
    v = str(v)
    if v in _VEHICLE_BADGE_TIERS:
        return _VEHICLE_BADGE_TIERS[v]
    if v.startswith("32 Ft"):
        return _VEHICLE_BADGE_TIERS["32 Ft"]
    return "background-color:#f1f5f9;color:#334155;font-weight:700;border-radius:6px"

def _shipment_heat_style(v, vmax):
    """Light-to-strong orange scale for a shipment-count column."""
    if vmax <= 0:
        return ""
    frac = min(1.0, v / vmax)
    if frac >= 0.66:
        return "background-color:#fed7aa;color:#7c2d12;font-weight:700"
    if frac >= 0.33:
        return "background-color:#ffedd5;color:#9a3412"
    return "background-color:#fff7ed;color:#9a3412"


def table_heading(text):
    """Big, centered heading used above each data table."""
    st.markdown(
        f'<h3 style="text-align:center;font-size:26px;font-weight:800;margin:6px 0 12px">{text}</h3>',
        unsafe_allow_html=True,
    )


def kcard(label, val, sub="", ac="#2563eb"):
    st.markdown(
        f'<div class="kcard" style="--ac:{ac}">'
        f'<div class="klabel">{label}</div>'
        f'<div class="kvalue">{val}</div>'
        f'{"<div class=ksub>"+sub+"</div>" if sub else ""}'
        f'</div>', unsafe_allow_html=True)

def pbar(label, val, cap, unit, color):
    pct = min(100, round(val / cap * 100, 1)) if cap else 0
    st.markdown(
        f'<div style="margin-bottom:12px">'
        f'<div style="display:flex;justify-content:space-between;font-size:13px">'
        f'<span style="font-weight:600;color:#334155">{label}</span>'
        f'<span style="color:#64748b">{val:,} / {cap:,} {unit} &nbsp;·&nbsp; <b>{pct}%</b></span>'
        f'</div>'
        f'<div class="bartrack"><div class="barfill" style="width:{pct}%;background:{color}"></div></div>'
        f'</div>', unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.markdown("## 🚛 Vehicle Predictor")
        st.caption("MotherHub_HJR")
        st.divider()
        st.markdown("""
**How to use**
1. Select cutoff(s) on the left, click **Confirm**
2. Select DH(s) on the right, click **Confirm**
3. Combined prediction appears **above** in the top box
4. Use **Vehicle Simulator** below to explore any size
""")
        st.divider()
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading data"):
        try:
            raw  = load_sheets()
            _key = tuple(sorted(raw.keys()))
            df_bag, df_semi, df_tote, df_sec, vcaps, df_dh = parse(_key)
        except Exception as e:
            st.error(f"❌ Could not load sheet: {e}")
            st.stop()

    max_cap = max(c for _, c in vcaps)

    with st.spinner("Computing DH loads…"):
        all_dh_loads = compute_all_dh_loads(df_bag, df_semi, df_tote, df_sec, df_dh)

    if df_dh.empty:
        st.warning("⚠️ DH Name Cut-Off sheet not found.")
        return

    # ── Confirmed (submitted) selections persist in session state ────────────────
    for k, v in [("sel_cutoffs", []), ("sel_dh_names", [])]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Single combined box: overall overview by default, swapped for the
    #    combined prediction once a DH selection is confirmed ────────────────────
    main_box = st.empty()

    def render_overview_default():
        bag_ships = df_bag["ship_count"].sum() if not df_bag.empty else 0
        main_box.markdown(
            f'<div class="predcard" style="display:flex;align-items:center;justify-content:space-around;gap:24px">'
            f'  <div style="text-align:center">'
            f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">🛍️ Total Bags on Floor</div>'
            f'    <div style="font-size:30px;font-weight:900;color:#f59e0b">{len(df_bag):,}</div>'
            f'    <div style="font-size:12px;opacity:.7">{bag_ships:,} shipments</div>'
            f'  </div>'
            f'  <div style="text-align:center;border-left:1px solid rgba(255,255,255,.25);padding-left:24px">'
            f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">📦 Semi-Large Shipments</div>'
            f'    <div style="font-size:30px;font-weight:900;color:#60a5fa">{len(df_semi):,}</div>'
            f'    <div style="font-size:12px;opacity:.7">Floor pending</div>'
            f'  </div>'
            f'  <div style="text-align:center;border-left:1px solid rgba(255,255,255,.25);padding-left:24px">'
            f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">🧺 Totes on Floor</div>'
            f'    <div style="font-size:30px;font-weight:900;color:#c4b5fd">{len(df_tote):,}</div>'
            f'    <div style="font-size:12px;opacity:.7">Pending dispatch</div>'
            f'  </div>'
            f'  <div style="text-align:center;border-left:1px solid rgba(255,255,255,.25);padding-left:24px">'
            f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">📋 Secondary + Bagging Pending</div>'
            f'    <div style="font-size:30px;font-weight:900;color:#fca5a5">{len(df_sec):,}</div>'
            f'    <div style="font-size:12px;opacity:.7">Sorted, not bagged</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    render_overview_default()
    st.divider()

    # ── Total pending shipments per cutoff (for the new cutoff-table column) ────
    cutoff_ship_totals = {}
    for _, dr in df_dh.drop_duplicates("dh_name").iterrows():
        dh_n = str(dr["dh_name"])
        co   = dr["cutoff_display"]
        ld   = all_dh_loads.get(dh_n, dict(bag_shipments=0, semi_count=0, tote_count=0, secondary_count=0))
        tot  = ld["bag_shipments"] + ld["semi_count"] + ld["tote_count"] + ld["secondary_count"]
        cutoff_ship_totals[co] = cutoff_ship_totals.get(co, 0) + tot

    cutoff_tbl = (
        df_dh.groupby("cutoff_display")
        .agg(DH_Count=("dh_name", "nunique"))
        .reset_index()
        .rename(columns={"cutoff_display": "Cutoff", "DH_Count": "# DHs"})
        .sort_values("Cutoff")
        .reset_index(drop=True)
    )
    cutoff_tbl["Total Shipment"] = cutoff_tbl["Cutoff"].map(cutoff_ship_totals).fillna(0).astype(int)

    # Table heights — DH table (right) is sized to match the combined height of
    # the cutoff table + all-vehicles table stacked in the left column.
    cutoff_h   = min(320, (len(cutoff_tbl) + 1) * 35 + 10)
    vehicles_h = min(380, (len(vcaps) + 1) * 35 + 10)
    dh_h       = cutoff_h + vehicles_h + 300

    # ── 2-column layout: Cutoff + All-Vehicles table (left) · DH table (right) ──
    col_left, col_right = st.columns([1, 1.6])

    cutoff_max_ship = int(cutoff_tbl["Total Shipment"].max()) if not cutoff_tbl.empty else 0
    cutoff_styled = cutoff_tbl.style.map(
        lambda v: _shipment_heat_style(v, cutoff_max_ship), subset=["Total Shipment"]
    )

    with col_left:
        table_heading("🕐 Select Cutoff(s)")
        with st.form("cutoff_form", border=False):
            cut_evt = st.dataframe(
                cutoff_styled,
                on_select="rerun",
                selection_mode="multi-row",
                use_container_width=True,
                hide_index=True,
                height=cutoff_h,
                column_config={
                    "Cutoff":         st.column_config.TextColumn(alignment="center"),
                    "# DHs":          st.column_config.NumberColumn(alignment="center", format="%d"),
                    "Total Shipment": st.column_config.NumberColumn(alignment="center", format="%d"),
                },
            )
            submitted_cut = st.form_submit_button("✅ Confirm Cutoff Selection", use_container_width=True)

        if submitted_cut:
            new_cutoffs = [cutoff_tbl.iloc[i]["Cutoff"] for i in cut_evt.selection.rows]
            if new_cutoffs != st.session_state.sel_cutoffs:
                st.session_state.sel_dh_names = []
            st.session_state.sel_cutoffs = new_cutoffs

        sel_cutoffs = st.session_state.sel_cutoffs

        table_heading("📋 All Vehicles — Comparison Table")
        vehicles_placeholder = st.empty()
        vehicles_placeholder.caption("Confirm a DH selection on the right to populate this table.")

    dh_loads_map = {}
    dh_summary   = pd.DataFrame()

    with col_right:
        if not sel_cutoffs:
            table_heading("🏭 DH Load Breakdown")
            st.info("👆 Select cutoff(s) on the left and click **Confirm** to see the DH breakdown.")
        else:
            filt_dh = df_dh[df_dh["cutoff_display"].isin(sel_cutoffs)].copy()
            dh_rows = []
            for _, dr in filt_dh.drop_duplicates("dh_name").iterrows():
                dh_n = str(dr["dh_name"])
                ld   = all_dh_loads.get(dh_n, dict(bag_count=0, bag_shipments=0,
                                                   semi_count=0, tote_count=0, secondary_count=0))
                dh_loads_map[dh_n] = ld

                sec_bags        = int(np.ceil(ld["secondary_count"] / SHIPMENTS_PER_BAG)) if ld["secondary_count"] > 0 else 0
                total_bags      = ld["bag_count"] + sec_bags
                total_bag_ships = ld["bag_shipments"] + ld["secondary_count"]
                total_ship_row  = total_bag_ships + ld["semi_count"] + ld["tote_count"]

                merged_ld = dict(bag_shipments=total_bag_ships, semi_count=ld["semi_count"],
                                 tote_count=ld["tote_count"], secondary_count=0)
                frac = load_to_frac(merged_ld)
                rec_v, rec_cap, rec_util, _, _ = recommend_vehicle(frac, vcaps)

                dh_rows.append({
                    "Cut Off":             dr["cutoff_display"],
                    "DH Code":             str(dr.get("dh_code", "")),
                    "DH Name":             dh_n,
                    "Bag":                 total_bags,
                    "Semi Large":          ld["semi_count"],
                    "Totes":               ld["tote_count"],
                    "Total Shipment":      total_ship_row,
                    "Max Vehicle Size":    rec_cap if rec_cap else 0,
                    "Recommended Vehicle": rec_v   if rec_v   else "—",
                    "Utilization %":       round(rec_util * 100, 1) if rec_v else 0.0,
                })

            dh_summary = pd.DataFrame(dh_rows)
            if not dh_summary.empty:
                dh_summary = (
                    dh_summary[dh_summary["Total Shipment"] > 0]
                    .sort_values(["Cut Off", "DH Name"])
                    .reset_index(drop=True)
                )
                dh_summary["Status"] = dh_summary["Utilization %"].apply(_status_dot)

            table_heading(f"🏭 DH Load Breakdown — {len(dh_summary)} DH(s) with pending load")

            if dh_summary.empty:
                st.success("✅ No pending floor load for any DH in the selected cutoff(s).")
            else:
                dh_styled = dh_summary.style.map(
                    _vehicle_badge_style, subset=["Recommended Vehicle"]
                )
                with st.form("dh_form", border=False):
                    dh_evt = st.dataframe(
                        dh_styled,
                        on_select="rerun",
                        selection_mode="multi-row",
                        use_container_width=True,
                        hide_index=True,
                        height=dh_h,
                        column_config={
                            "Cut Off":             st.column_config.TextColumn(alignment="center"),
                            "DH Code":             st.column_config.TextColumn(alignment="center"),
                            "DH Name":             st.column_config.TextColumn(alignment="center"),
                            "Bag":                 st.column_config.NumberColumn(alignment="center", format="%d"),
                            "Semi Large":          st.column_config.NumberColumn(alignment="center", format="%d"),
                            "Totes":               st.column_config.NumberColumn(alignment="center", format="%d"),
                            "Total Shipment":      st.column_config.NumberColumn(alignment="center", format="%d"),
                            "Max Vehicle Size":    st.column_config.NumberColumn(alignment="center", format="%d"),
                            "Recommended Vehicle": st.column_config.TextColumn(alignment="center"),
                            "Utilization %":       st.column_config.ProgressColumn(
                                format="%.1f%%", min_value=0, max_value=100,
                            ),
                            "Status":              st.column_config.TextColumn(alignment="center", width="small"),
                        },
                    )
                    submitted_dh = st.form_submit_button("✅ Confirm DH Selection", use_container_width=True)

                if submitted_dh:
                    st.session_state.sel_dh_names = [dh_summary.iloc[i]["DH Name"] for i in dh_evt.selection.rows]

    sel_names = [n for n in st.session_state.sel_dh_names if n in dh_loads_map]

    # ── Aggregate confirmed DH selection (secondary already folded into bags) ──
    agg = dict(bag_count=0, bag_shipments=0, semi_count=0, tote_count=0, secondary_count=0)
    for dh_n in sel_names:
        ld = dh_loads_map[dh_n]
        sec_bags = int(np.ceil(ld["secondary_count"] / SHIPMENTS_PER_BAG)) if ld["secondary_count"] > 0 else 0
        agg["bag_count"]     += ld["bag_count"] + sec_bags
        agg["bag_shipments"] += ld["bag_shipments"] + ld["secondary_count"]
        agg["semi_count"]    += ld["semi_count"]
        agg["tote_count"]    += ld["tote_count"]

    total_ship = agg["bag_shipments"] + agg["semi_count"] + agg["tote_count"]
    total_frac = load_to_frac(agg) if total_ship else 0.0
    req_equiv  = frac_to_equiv(total_frac, max_cap) if total_ship else 0
    best_v = best_cap = best_util = n_trucks = None

    if total_ship:
        best_v, best_cap, best_util, n_trucks, truck_breakdown = recommend_vehicle(total_frac, vcaps)

    # ── Fill the All Vehicles comparison table (left column) ───────────────────
    # ship_per_equiv converts the normalized "equivalent" capacity unit back
    # into the real shipment count for the actual bag/semi/tote/secondary mix
    # selected, so the table shows exact shipments — never the equivalent unit.
    ship_per_equiv = (total_ship / req_equiv) if req_equiv else 0

    veh_rows = []
    for v, c in vcaps:
        f    = min(req_equiv, c)
        u    = round(f / c * 100, 1) if c else 0
        left = max(0, req_equiv - c)
        r90  = max(0, c * TARGET_UTIL - f)
        veh_rows.append({
            "Vehicle": v,
            "Capacity": f"{c:,}",
            "Can Load": f"{int(round(f * ship_per_equiv)):,}",
            "Utilization": u,
            "Status": _status_dot(u),
            "Remains on Floor": f"{int(round(left * ship_per_equiv)):,}",
            "More to reach 100%": f"{int(round(r90 * ship_per_equiv)):,}" if r90 > 0 else "✅ Optimal",
            "Trucks Needed": int(np.ceil(req_equiv / c)) if c and req_equiv else 0,
        })
    veh_df = pd.DataFrame(veh_rows)
    veh_styled = veh_df.style.map(_vehicle_badge_style, subset=["Vehicle"])
    vehicles_placeholder.dataframe(
        veh_styled, use_container_width=True, hide_index=True,
        height=vehicles_h,
        column_config={
            "Vehicle":              st.column_config.TextColumn(alignment="center"),
            "Capacity":             st.column_config.TextColumn(alignment="center"),
            "Can Load":             st.column_config.TextColumn(alignment="center"),
            "Utilization":          st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Status":               st.column_config.TextColumn(alignment="center", width="small"),
            "Remains on Floor":     st.column_config.TextColumn(alignment="center"),
            "More to reach 100%":   st.column_config.TextColumn(alignment="center"),
            "Trucks Needed":        st.column_config.NumberColumn(alignment="center", format="%d"),
        },
    )

    # ── Fill the combined box with the prediction, once a DH selection is confirmed ──
    if best_v is not None:
        util_pct = round(best_util * 100, 1)
        conf_col = "#16a34a" if util_pct >= 75 else "#f59e0b" if util_pct >= 40 else "#ef4444"

        # Utilization block: one blended number for a single vehicle, or one
        # line PER TRUCK when the load spans multiple trucks — each truck's
        # own utilization shown individually, even if the vehicle type repeats.
        if len(truck_breakdown) > 1:
            util_lines = ""
            for i, tb in enumerate(truck_breakdown, start=1):
                pct    = round(tb["util_frac"] * 100, 1)
                tb_col = "#4ade80" if pct >= 75 else "#fbbf24" if pct >= 40 else "#f87171"
                util_lines += (
                    f'<div style="font-size:14px;font-weight:800;margin-top:4px;color:{tb_col}">'
                    f'Truck {i} ({tb["vehicle"]}): {pct}%</div>'
                )
            util_block = (
                f'<div style="text-align:center;border-left:1px solid rgba(255,255,255,.25);padding-left:24px">'
                f'<div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">Utilization / Vehicle</div>'
                f'{util_lines}'
                f'</div>'
            )
        else:
            util_block = (
                f'<div style="text-align:center;border-left:1px solid rgba(255,255,255,.25);padding-left:24px">'
                f'<div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">Load Utilization</div>'
                f'<div style="font-size:28px;font-weight:900;color:{conf_col}">{util_pct}%</div>'
                f'</div>'
            )

        with main_box.container():
            st.markdown(
                f'<div class="predcard" style="display:flex;align-items:center;justify-content:space-between;gap:24px">'
                # ── LEFT: load bifurcation for the confirmed selection ──
                f'<div style="flex:1;min-width:0">'
                f'  <div style="font-size:13px;opacity:.8;font-weight:500">📦 Load Bifurcation — {len(sel_names)} DH(s)</div>'
                f'  <div style="font-size:14px;margin-top:8px;line-height:1.9">'
                f'    🛍️ <b>{agg["bag_count"]:,}</b> bags &nbsp;({agg["bag_shipments"]:,} shipments)<br>'
                f'    📦 <b>{agg["semi_count"]:,}</b> semi-large shipments<br>'
                f'    🧺 <b>{agg["tote_count"]:,}</b> totes'
                f'  </div>'
                f'</div>'
                # ── RIGHT: vehicle prediction ──
                f'<div style="display:flex;align-items:center;gap:28px;flex-shrink:0;border-left:1px solid rgba(255,255,255,.25);padding-left:28px">'
                f'  <div>'
                f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">🎯 Recommended Vehicle</div>'
                f'    <div style="font-size:42px;font-weight:900;margin:2px 0;letter-spacing:-1px">{best_v}</div>'
                f'  </div>'
                f'{util_block}'
                f'  <div style="text-align:center;border-left:1px solid rgba(255,255,255,.25);padding-left:24px">'
                f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">Trucks Needed</div>'
                f'    <div style="font-size:28px;font-weight:900">{n_trucks}</div>'
                f'  </div>'
                f'  <div style="text-align:center;border-left:1px solid rgba(255,255,255,.25);padding-left:24px">'
                f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">Total Shipments</div>'
                f'    <div style="font-size:28px;font-weight:900">{total_ship:,}</div>'
                f'  </div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # DH names as wrapping chips BELOW the box (outside, one after another)
            st.markdown(f"#### Selected DHs ({len(sel_names)})")
            chips = "".join(
                f'<span style="display:inline-block;background:#eef2ff;color:#3730a3;'
                f'border-radius:6px;padding:3px 10px;margin:4px 4px 0 0;font-size:12px;font-weight:600">{n}</span>'
                for n in sel_names
            )
            st.markdown(f'<div style="line-height:2.2">{chips}</div>', unsafe_allow_html=True)
    elif sel_names:
        with main_box.container():
            st.success(f"✅ No pending floor load for {len(sel_names)} selected DH(s).")

    st.divider()

    if best_v is None:
        st.info("👆 Select cutoff(s), click Confirm, then select DH(s) on the right and click Confirm to see the combined prediction above.")
        return

    st.markdown("### 🔧 Vehicle Load Simulator")

    vnames = [v for v, _ in vcaps]
    sel_v  = st.selectbox(
        "Select Vehicle Size",
        vnames,
        index=next((i for i, (v, _) in enumerate(vcaps) if v == best_v), 0)
        if best_v and best_v in vnames else 0,
    )

    sel_cap      = next(c for v, c in vcaps if v == sel_v)
    load_eq      = req_equiv   

    bag_cap_v  = sel_cap / max_cap * BAG_SHIPMENTS_32FT   
    semi_cap_v = sel_cap / max_cap * SEMI_32FT
    tote_cap_v = sel_cap / max_cap * TOTES_32FT
    sec_cap_v  = sel_cap / max_cap * SECONDARY_32FT

    act_bag_ships = agg["bag_shipments"]
    act_bag_count = agg["bag_count"]
    act_semi      = agg["semi_count"]
    act_totes     = agg["tote_count"]
    act_sec       = agg["secondary_count"]

    can_fit_eq  = min(load_eq, sel_cap)
    cant_fit_eq = max(0.0, load_eq - sel_cap)
    all_fit     = cant_fit_eq == 0

    if all_fit:
        can_bag_ships = act_bag_ships
        can_bags      = act_bag_count
        can_semi      = act_semi
        can_totes     = act_totes
        can_sec       = act_sec
        rem_bag_ships = 0; rem_bags = 0; rem_semi = 0; rem_totes = 0; rem_sec = 0
    else:
        fit_ratio     = can_fit_eq / load_eq if load_eq else 0
        can_bag_ships = int(act_bag_ships * fit_ratio)
        can_bags      = int(act_bag_count * fit_ratio)
        can_semi      = int(act_semi  * fit_ratio)
        can_totes     = int(act_totes * fit_ratio)
        can_sec       = int(act_sec   * fit_ratio)
        rem_bag_ships = act_bag_ships - can_bag_ships
        rem_bags      = act_bag_count - can_bags
        rem_semi      = act_semi  - can_semi
        rem_totes     = act_totes - can_totes
        rem_sec       = act_sec   - can_sec

    util_sel = round(can_fit_eq / sel_cap * 100, 1) if sel_cap else 0
    util_col  = "#16a34a" if util_sel >= 80 else "#f59e0b" if util_sel >= 50 else "#ef4444"

    rem_to_90_eq = remaining_to_target(can_fit_eq, sel_cap, max_cap)
    rem_to_90_bd = breakdown_remaining(rem_to_90_eq, max_cap)

    # Trucks needed of the SELECTED vehicle size to clear the whole floor load
    # (not the 32 Ft-based count — this depends on which size is picked above).
    veh_trucks_needed = int(np.ceil(load_eq / sel_cap)) if sel_cap and load_eq else 1

    r1, r2, r3 = st.columns([1, 1, 0.6])
    with r1:
        st.markdown(
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px">'
            f'<div class="klabel">🚛 {sel_v} Capacity</div>'
            f'<div style="font-size:28px;font-weight:800;color:#1e293b;margin:4px 0">'
            f'{sel_cap:,} <span style="font-size:14px;color:#64748b">shipments</span></div>'
            f'<hr style="border:none;border-top:1px solid #f1f5f9;margin:10px 0">'
            f'<div class="klabel">Current Utilization</div>'
            f'<div style="font-size:32px;font-weight:900;color:{util_col}">{util_sel}%</div>'
            f'<div style="margin-top:10px">'
            f'<div class="bartrack"><div class="barfill" style="width:{util_sel}%;background:{util_col}"></div></div>'
            f'</div>'
            f'<div style="font-size:13px;color:#64748b;margin-top:10px">'
            f'Loading <b>{can_bag_ships + can_semi + can_totes + can_sec:,}</b> of <b>{total_ship:,}</b> total shipments</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px">'
            f'<div class="klabel" style="color:#16a34a">✅ Loads in {sel_v} — Actual Floor Data</div>'
            f'<div style="font-size:13px;margin:8px 0 12px">'
            f'&nbsp;🛍️ <b>{can_bags:,}</b> bags &nbsp;({can_bag_ships:,} shipments)<br>'
            f'&nbsp;📦 <b>{can_semi:,}</b> semi-large shipments<br>'
            f'&nbsp;🧺 <b>{can_totes:,}</b> totes<br>'
            f'&nbsp;📋 <b>{can_sec:,}</b> secondary shipments'
            f'</div>'
            f'<hr style="border:none;border-top:1px solid #f1f5f9;margin:0 0 10px">'
            f'<div class="klabel" style="color:#ef4444">⏳ Remains on Floor</div>'
            f'<div style="font-size:13px;margin:8px 0">'
            f'&nbsp;🛍️ <b>{rem_bags:,}</b> bags &nbsp;({rem_bag_ships:,} shipments)<br>'
            f'📦 <b>{rem_semi:,}</b> semi-large &nbsp;|&nbsp; '
            f'🧺 <b>{rem_totes:,}</b> totes &nbsp;|&nbsp; '
            f'📋 <b>{rem_sec:,}</b> secondary'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with r3:
        kcard("🚚 Trucks Needed", f"{veh_trucks_needed}", f"of {sel_v} to clear the floor", "#7c3aed")

    if rem_to_90_eq > 0:
        st.markdown(
            f"<div style='background:#fefce8;border:1px solid #fde047;border-radius:10px;"
            f"padding:12px 16px;font-size:13px;margin-top:8px'>"
            f"<b>📈 To reach 100% in {sel_v} ({int(sel_cap*TARGET_UTIL):,} shipments):</b>&nbsp; "
            f"Can load &nbsp;"
            f"<b>{rem_to_90_bd['bags']:,}</b> bags &nbsp;|&nbsp; "
            f"<b>{rem_to_90_bd['semi']:,}</b> semi-large &nbsp;|&nbsp; "
            f"<b>{rem_to_90_bd['totes']:,}</b> totes"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 🧪 Load Mix Simulator")
    st.caption(f"Select which load types to dispatch. Vehicle **{sel_v}** (selected above) is used to calculate utilization.")

    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        inc_bag = st.checkbox(
            f"🛍️ Bags  ({agg['bag_count']:,} bags · {agg['bag_shipments']:,} shipments)",
            value=True,
        )
    with lc2:
        inc_semi = st.checkbox(
            f"📦 Semi-Large  ({agg['semi_count']:,} shipments)",
            value=True,
        )
    with lc3:
        inc_tote = st.checkbox(
            f"🧺 Totes  ({agg['tote_count']:,} totes)",
            value=True,
        )
    with lc4:
        inc_sec = st.checkbox(
            f"📋 Secondary + Bagging  ({agg['secondary_count']:,} shipments)",
            value=True,
        )

    mix = dict(
        bag_shipments   = agg["bag_shipments"]    if inc_bag  else 0,
        bag_count       = agg["bag_count"]         if inc_bag  else 0,
        semi_count      = agg["semi_count"]        if inc_semi else 0,
        tote_count      = agg["tote_count"]        if inc_tote else 0,
        secondary_count = agg["secondary_count"]   if inc_sec  else 0,
    )

    mix_total_ship = mix["bag_shipments"] + mix["semi_count"] + mix["tote_count"] + mix["secondary_count"]
    mix_frac       = load_to_frac(mix)
    mix_equiv      = frac_to_equiv(mix_frac, max_cap)

    # What the selected vehicle can actually hold vs. what overflows —
    # a single vehicle can never be loaded beyond its own capacity, so any
    # excess must show up as pending-on-floor instead of >100% utilization.
    mix_can_fit_eq  = min(mix_equiv, sel_cap)
    mix_cant_fit_eq = max(0.0, mix_equiv - sel_cap)
    mix_all_fit     = mix_cant_fit_eq == 0

    if mix_all_fit:
        mix_can_bag_ships = mix["bag_shipments"]
        mix_can_bags      = mix["bag_count"]
        mix_can_semi      = mix["semi_count"]
        mix_can_totes     = mix["tote_count"]
        mix_can_sec       = mix["secondary_count"]
        overflow_bag_ships = overflow_bags = overflow_semi = overflow_totes = overflow_sec = 0
    else:
        fit_ratio = mix_can_fit_eq / mix_equiv if mix_equiv else 0
        mix_can_bag_ships = int(mix["bag_shipments"] * fit_ratio)
        mix_can_bags      = int(mix["bag_count"] * fit_ratio)
        mix_can_semi      = int(mix["semi_count"] * fit_ratio)
        mix_can_totes     = int(mix["tote_count"] * fit_ratio)
        mix_can_sec       = int(mix["secondary_count"] * fit_ratio)
        overflow_bag_ships = mix["bag_shipments"]    - mix_can_bag_ships
        overflow_bags      = mix["bag_count"]         - mix_can_bags
        overflow_semi      = mix["semi_count"]        - mix_can_semi
        overflow_totes     = mix["tote_count"]        - mix_can_totes
        overflow_sec       = mix["secondary_count"]   - mix_can_sec

    mix_loading_ship   = mix_can_bag_ships + mix_can_semi + mix_can_totes + mix_can_sec
    overflow_total_real = overflow_bag_ships + overflow_semi + overflow_totes + overflow_sec

    floor_bag_ships = (agg["bag_shipments"]  - mix["bag_shipments"]) + overflow_bag_ships
    floor_bags      = (agg["bag_count"]       - mix["bag_count"])     + overflow_bags
    floor_semi      = (agg["semi_count"]      - mix["semi_count"])    + overflow_semi
    floor_totes     = (agg["tote_count"]      - mix["tote_count"])    + overflow_totes
    floor_sec       = (agg["secondary_count"] - mix["secondary_count"]) + overflow_sec
    floor_total     = floor_bag_ships + floor_semi + floor_totes + floor_sec

    if mix_total_ship == 0:
        st.warning("⚠️ No load types selected. Select at least one type above.")
    else:
        mix_v    = sel_v
        mix_cap  = sel_cap
        mix_util = mix_can_fit_eq / sel_cap if sel_cap else 0
        mix_trucks = max(1, int(np.ceil(mix_equiv / sel_cap))) if sel_cap else 1
        mix_util_pct = round(mix_util * 100, 1)
        mix_col      = "#16a34a" if mix_util_pct >= 75 else "#f59e0b" if mix_util_pct >= 40 else "#ef4444"

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        with sm1: kcard("🚛 Predicted Vehicle",    str(mix_v),              "for selected mix", "#2563eb")
        with sm2: kcard("📊 Load Utilization",     f"{mix_util_pct}%",      "of vehicle capacity", mix_col)
        with sm3: kcard("📦 Loading",              f"{mix_loading_ship:,}", "shipments in this mix", "#16a34a")
        with sm4: kcard("⏳ Remains on Floor",     f"{floor_total:,}",      "shipments not dispatched", "#ef4444")
        with sm5: kcard("🚚 Trucks Needed",        f"{mix_trucks}",         f"of {mix_v} for this mix", "#7c3aed")

        if not mix_all_fit:
            st.warning(
                f"⚠️ Selected mix ({mix_total_ship:,} shipments) exceeds {mix_v}'s capacity "
                f"({sel_cap:,}) — {overflow_total_real:,} shipments can't fit and stay on the floor."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            mix_util_bar = min(mix_util_pct, 100)
            mix_rem_eq   = remaining_to_target(mix_equiv, mix_cap if isinstance(mix_cap, int) else max_cap, max_cap)
            mix_rem_bd   = breakdown_remaining(mix_rem_eq, max_cap)

            mix_bag_ships_fmt = f"{mix['bag_shipments']:,}"
            d1_lines = ""
            if inc_bag:
                d1_lines += f"🛍️ <b>{mix['bag_count']}</b> bags ({mix_bag_ships_fmt} shipments)<br>"
            if inc_semi:
                d1_lines += f"📦 <b>{mix['semi_count']}</b> semi-large shipments<br>"
            if inc_tote:
                d1_lines += f"🧺 <b>{mix['tote_count']}</b> totes<br>"
            if inc_sec:
                d1_lines += f"📋 <b>{mix['secondary_count']}</b> secondary shipments<br>"

            rem_hint_color = "#64748b"
            if mix_rem_eq > 0:
                rem_hint = (
                    f'<div style="font-size:12px;color:{rem_hint_color};margin-top:8px">'
                    f'📈 To reach 100%, can add: {mix_rem_bd["bags"]} bags | {mix_rem_bd["semi"]} semi'
                    f' | {mix_rem_bd["totes"]} totes</div>'
                )
            else:
                rem_hint = '<div style="font-size:12px;color:#16a34a;margin-top:8px">✅ Optimal — 100% utilized</div>'

            st.markdown(
                f'<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px">'
                f'<div class="klabel">🚛 {mix_v} — Selected Mix Detail</div>'
                f'<div style="font-size:13px;margin:10px 0 6px">{d1_lines}</div>'
                f'<div class="klabel" style="margin-top:10px">Utilization</div>'
                f'<div style="font-size:28px;font-weight:900;color:{mix_col}">{mix_util_pct}%</div>'
                f'<div class="bartrack" style="margin-top:6px">'
                f'<div class="barfill" style="width:{mix_util_bar}%;background:{mix_col}"></div></div>'
                f'{rem_hint}'
                f'</div>',
                unsafe_allow_html=True,
            )

        with d2:
            floor_color = "#ef4444" if floor_total > 0 else "#16a34a"
            floor_bag_ships_fmt = f"{floor_bag_ships:,}"
            d2_lines = ""
            if floor_bags > 0 or not inc_bag:
                d2_lines += f"🛍️ <b>{floor_bags}</b> bags ({floor_bag_ships_fmt} shipments)<br>"
            if floor_semi > 0 or not inc_semi:
                d2_lines += f"📦 <b>{floor_semi}</b> semi-large shipments<br>"
            if floor_totes > 0 or not inc_tote:
                d2_lines += f"🧺 <b>{floor_totes}</b> totes<br>"
            if floor_sec > 0 or not inc_sec:
                d2_lines += f"📋 <b>{floor_sec}</b> secondary shipments<br>"
            if floor_total == 0:
                d2_lines = "✅ <b>All selected types dispatched — nothing remains on floor</b>"

            st.markdown(
                f'<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px">'
                f'<div class="klabel" style="color:#ef4444">⏳ Remains on Floor (not dispatched)</div>'
                f'<div style="font-size:13px;margin:10px 0">{d2_lines}</div>'
                f'<hr style="border:none;border-top:1px solid #f1f5f9;margin:10px 0">'
                f'<div class="klabel">Total Remaining</div>'
                f'<div style="font-size:28px;font-weight:900;color:{floor_color}">'
                f'{floor_total:,} <span style="font-size:14px;color:#64748b">shipments</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if mix_total_ship < total_ship:
            st.markdown(
                f"<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;"
                f"padding:12px 16px;font-size:13px;margin-top:8px'>"
                f"<b>📊 vs Full Load:</b> &nbsp; "
                f"Dispatching <b>{mix_total_ship:,}</b> of <b>{total_ship:,}</b> shipments "
                f"(<b>{round(mix_total_ship/total_ship*100,1)}%</b> of floor load). &nbsp;"
                f"Full load needs: <b>{best_v}</b> at <b>{util_pct}%</b> utilization."
                f"</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
