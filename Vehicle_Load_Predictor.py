"""
Vehicle Load Prediction Dashboard  —  Flipkart · Hajipur Mother Hub
• Multi-select cutoff + DH tables
• Smart vehicle recommendation targeting 100 % utilisation
• Manual vehicle selector with remaining-capacity breakdown
"""

import re
import warnings
from datetime import datetime
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

SIDEBAR_WIDTH_PX = 190

st.markdown("""
<style>
/* Diagonal "N K" watermark tiled across the page background — sits behind
   all content (it's part of .stApp's own background paint, so any card/
   table/box drawn on top of it naturally covers it — never overlaps).
   Very faint at normal zoom, reads clearly when zoomed in. */
.stApp{
    background-color:#f0f2f6;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='260' height='200'%3E%3Ctext x='20' y='110' font-family='Arial,sans-serif' font-size='36' font-weight='800' fill='rgba(30,41,59,0.025)' transform='rotate(-30 130 100)'%3EN K%3C/text%3E%3C/svg%3E");
    background-repeat:repeat;
}
/* Sidebar: narrow, always-expanded nav rail holding just the two tab
   buttons — no collapse arrow, fixed minimum width so it doesn't eat
   dashboard space. */
section[data-testid="stSidebar"]{
    width:190px!important; min-width:190px!important; max-width:190px!important;
}
button[data-testid="stSidebarCollapseButton"]{display:none!important}
div[data-testid="stSidebarResizeHandle"]{display:none!important}
/* Reduce the default top padding Streamlit adds below the sticky header */
div[data-testid="stAppViewContainer"] .block-container{padding-top:72px!important}
/* Combined Prediction box + Selected DHs pinned to the top while scrolling.
   NOTE: position:sticky does NOT work here — Streamlit wraps every
   st.container() in its own shrink-to-fit wrapper div, which never gives a
   sticky child room to "stick" within (verified via direct DOM testing:
   the wrapper's height always equals its child's height, so there's no
   scrollable range for sticky to hold position against). Using
   position:fixed instead, which is proven to work in this app already
   (header credit, Refresh button). left is offset past the sidebar's
   width so the box doesn't render on top of/underneath it. */
.st-key-pred_sticky, .st-key-ready_pred_sticky{
    position:fixed!important; top:80px!important; left:214px!important; right:24px!important;
    width:calc(100vw - 238px)!important; max-width:calc(100vw - 238px)!important;
    flex:none!important; box-sizing:border-box!important; overflow-x:auto;
    z-index:500; background:#f0f2f6; padding-bottom:10px;
}
/* Reserves the space the box would have occupied in normal flow, since
   position:fixed removes it — otherwise content below jumps up underneath it. */
.pred-sticky-spacer{height:180px}
@media (max-width:900px){
    .pred-sticky-spacer{height:260px}
}
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
/* Small developer credit — pinned just BELOW the header's toolbar row
   (Stop/Share/Star/Edit/GitHub/menu) so it never overlaps those controls,
   including while the app is running and shows the "Stop" indicator. */
.dev-credit{
    position:fixed; top:58px; right:16px; z-index:999999;
    font-size:11px; font-weight:500; color:#94a3b8; white-space:nowrap;
    pointer-events:none; background:#f0f2f6;
}
@media (max-width:700px){
    .dev-credit{display:none}
}

/* Sidebar buttons (nav + refresh): full width, consistent sizing */
section[data-testid="stSidebar"] div[data-testid="stButton"] button{
    font-size:13px!important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dev-credit">Developed by Nitesh Kumar</div>', unsafe_allow_html=True)

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

@st.cache_data(ttl=300, show_spinner=False)
def _data_fetched_at(_key):
    """Timestamp of the last actual sheet fetch (shares load_sheets' cache
    lifecycle via the same _key/ttl), for the 'data last updated' display."""
    return datetime.now()

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

    vehcap_v = _find(sheets, "vehicle capacity")
    dh_max_vehicle = {}
    if vehcap_v:
        d = _df(vehcap_v)
        name_col = next((c for c in d.columns if c.strip().lower() == "dh name"), None)
        size_col = next((c for c in d.columns if c.strip().lower() == "vehicle size"), None)
        if name_col and size_col:
            for _, row in d.iterrows():
                nm = str(row[name_col]).strip()
                sz = str(row[size_col]).strip()
                if nm and nm.lower() != "nan" and sz and sz.lower() != "nan":
                    dh_max_vehicle.setdefault(_norm(nm), sz)

    return df_bag, df_semi, df_tote, df_sec, vcaps, df_dh, dh_max_vehicle


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

def _vehicle_size_num(vehicle_name):
    """Leading number from a vehicle label, e.g. '14 Ft' -> 14.0, '6.5 Ft' -> 6.5."""
    m = re.search(r"(\d+(?:\.\d+)?)", str(vehicle_name))
    return float(m.group(1)) if m else None

def allowed_vcaps_for(max_size_str, vcaps):
    """
    Restrict vcaps to only vehicles permitted for a DH, per the Vehicle
    Capacity sheet's max size (e.g. "14 Ft" excludes 17/20/22/24/32 Ft).
    No restriction (max_size_str falsy, or size not recognised) -> all vcaps.
    """
    if not max_size_str:
        return vcaps
    max_num = _vehicle_size_num(max_size_str)
    if max_num is None:
        return vcaps
    allowed = [(v, c) for v, c in vcaps if (_vehicle_size_num(v) or 0) <= max_num + 1e-6]
    return allowed if allowed else vcaps

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
        rem_frac       = total_frac - (n - 1)
        rem_cap        = rem_frac * max_cap
        last_v, last_c = next(((v, c) for v, c in vcaps if c >= rem_cap), vcaps[-1])
        last_util      = rem_cap / last_c if last_c else 0.0
        breakdown = [{"vehicle": max_v, "capacity": max_cap, "util_frac": 1.0} for _ in range(n - 1)]
        breakdown.append({"vehicle": last_v, "capacity": last_c, "util_frac": last_util})
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


def build_dh_rows(dh_source_df, all_dh_loads, dh_max_vehicle, vcaps):
    """
    Per-DH breakdown (Bag/Semi/Totes/Total Shipment/Max Vehicle Size/
    Recommended Vehicle/Utilization %) for every DH in dh_source_df — shared
    by the main tab (cutoff-filtered) and the Ready to Dispatch tab (all DHs).
    Returns (dh_summary_df, dh_loads_map).
    """
    dh_rows = []
    dh_loads_map = {}
    for _, dr in dh_source_df.drop_duplicates("dh_name").iterrows():
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

        max_v_str = dh_max_vehicle.get(_norm(dh_n))
        dh_vcaps  = allowed_vcaps_for(max_v_str, vcaps)
        rec_v, rec_cap, rec_util, _, _ = recommend_vehicle(frac, dh_vcaps)

        dh_rows.append({
            "Cut Off":             dr["cutoff_display"],
            "DH Code":             str(dr.get("dh_code", "")),
            "DH Name":             dh_n,
            "Bag":                 total_bags,
            "Semi Large":          ld["semi_count"],
            "Totes":               ld["tote_count"],
            "Total Shipment":      total_ship_row,
            "Max Vehicle Size":    max_v_str if max_v_str else "All vehicles",
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
    return dh_summary, dh_loads_map


def agg_for(names, dh_loads_map):
    """Aggregate raw per-DH loads for a list of DH names (secondary folded into bags)."""
    a = dict(bag_count=0, bag_shipments=0, semi_count=0, tote_count=0, secondary_count=0)
    for dh_n in names:
        ld = dh_loads_map[dh_n]
        sec_bags = int(np.ceil(ld["secondary_count"] / SHIPMENTS_PER_BAG)) if ld["secondary_count"] > 0 else 0
        a["bag_count"]     += ld["bag_count"] + sec_bags
        a["bag_shipments"] += ld["bag_shipments"] + ld["secondary_count"]
        a["semi_count"]    += ld["semi_count"]
        a["tote_count"]    += ld["tote_count"]
    return a


def render_prediction_box(main_box, sel_names, dh_loads_map, dh_max_vehicle, vcaps, max_cap):
    """
    Renders the combined Load Bifurcation + Recommended Vehicle prediction
    card into main_box (an st.empty()) for the given selected DH names.
    Shared by the main tab and the Ready to Dispatch tab.
    """
    agg = agg_for(sel_names, dh_loads_map)
    total_ship = agg["bag_shipments"] + agg["semi_count"] + agg["tote_count"]
    total_frac = load_to_frac(agg) if total_ship else 0.0
    req_equiv  = frac_to_equiv(total_frac, max_cap) if total_ship else 0
    best_v = best_cap = best_util = n_trucks = None
    truck_breakdown = []

    # Selected DHs may have DIFFERENT max-permissible-vehicle constraints —
    # the combined prediction is capped at the single most restrictive one.
    constrained_nums = [
        n for n in (
            _vehicle_size_num(dh_max_vehicle[_norm(dh_n)])
            for dh_n in sel_names
            if dh_max_vehicle.get(_norm(dh_n))
        ) if n is not None
    ]
    min_constraint = min(constrained_nums) if constrained_nums else None
    allowed = allowed_vcaps_for(f"{min_constraint} Ft" if min_constraint is not None else None, vcaps)

    if total_ship:
        best_v, best_cap, best_util, n_trucks, truck_breakdown = recommend_vehicle(total_frac, allowed)

    ship_per_equiv = (total_ship / req_equiv) if req_equiv else 0

    if best_v is not None:
        util_pct = round(best_util * 100, 1)
        conf_col = "#16a34a" if util_pct >= 75 else "#f59e0b" if util_pct >= 40 else "#ef4444"

        best_real_cap = int(round(best_cap * ship_per_equiv)) if isinstance(best_cap, int) else None

        if len(truck_breakdown) > 1:
            util_lines = ""
            for i, tb in enumerate(truck_breakdown, start=1):
                pct     = round(tb["util_frac"] * 100, 1)
                tb_col  = "#4ade80" if pct >= 75 else "#fbbf24" if pct >= 40 else "#f87171"
                tb_real = int(round(tb["capacity"] * ship_per_equiv))
                util_lines += (
                    f'<div style="font-size:14px;font-weight:800;margin-top:4px;color:{tb_col}">'
                    f'Truck {i} ({tb["vehicle"]}): {pct}%</div>'
                    f'<div style="font-size:11px;opacity:.7">~{tb_real:,} shipments capacity</div>'
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

        cap_sub = f'<div style="font-size:12px;opacity:.7;margin-top:4px">~{best_real_cap:,} shipments capacity</div>' if best_real_cap is not None else ""

        right_html = (
            f'<div style="display:flex;align-items:center;gap:28px;flex-shrink:0;border-left:1px solid rgba(255,255,255,.25);padding-left:28px">'
            f'  <div>'
            f'    <div style="font-size:11px;opacity:.75;font-weight:700;text-transform:uppercase;letter-spacing:.6px">🎯 Recommended Vehicle</div>'
            f'    <div style="font-size:42px;font-weight:900;margin:2px 0;letter-spacing:-1px">{best_v}</div>'
            f'    {cap_sub}'
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
        )

        with main_box.container():
            st.markdown(
                f'<div class="predcard" style="display:flex;align-items:center;justify-content:space-between;gap:24px">'
                f'<div style="flex:1;min-width:0">'
                f'  <div style="font-size:13px;opacity:.8;font-weight:500">📦 Load Bifurcation — {len(sel_names)} DH(s)</div>'
                f'  <div style="font-size:14px;margin-top:8px;line-height:1.9">'
                f'    🛍️ <b>{agg["bag_count"]:,}</b> bags &nbsp;({agg["bag_shipments"]:,} shipments)<br>'
                f'    📦 <b>{agg["semi_count"]:,}</b> semi-large shipments<br>'
                f'    🧺 <b>{agg["tote_count"]:,}</b> totes'
                f'  </div>'
                f'</div>'
                f'{right_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
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


def main():
    with st.spinner("Loading data"):
        try:
            raw  = load_sheets()
            _key = tuple(sorted(raw.keys()))
            df_bag, df_semi, df_tote, df_sec, vcaps, df_dh, dh_max_vehicle = parse(_key)
        except Exception as e:
            st.error(f"❌ Could not load sheet: {e}")
            st.stop()

    max_cap = max(c for _, c in vcaps)

    with st.spinner("Computing DH loads…"):
        all_dh_loads = compute_all_dh_loads(df_bag, df_semi, df_tote, df_sec, df_dh)

    if df_dh.empty:
        st.warning("⚠️ DH Name Cut-Off sheet not found.")
        return

    for k, v in [("sel_cutoffs", []), ("sel_dh_names", []), ("ready_sel_dh_names", [])]:
        if k not in st.session_state:
            st.session_state[k] = v

    last_updated = _data_fetched_at(_key)
    dh_h = 650

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "overview"

    with st.sidebar:
        if st.button(
            "📊 Overview", use_container_width=True,
            type="primary" if st.session_state.active_tab == "overview" else "secondary",
        ):
            st.session_state.active_tab = "overview"
        if st.button(
            "🚀 Ready to Dispatch", use_container_width=True,
            type="primary" if st.session_state.active_tab == "ready" else "secondary",
        ):
            st.session_state.active_tab = "ready"
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Tab 1: existing dashboard, unchanged ────────────────────────────────
    if st.session_state.active_tab == "overview":
        # Fixed (not sticky — see CSS comment) so this stays pinned below the
        # header while the cutoff dropdown / DH table scroll underneath it.
        with st.container(key="pred_sticky"):
            st.caption(f"📅 Data last updated: {last_updated.strftime('%d %b %Y, %I:%M %p')}")
            main_box = st.empty()
        # Reserves the vertical space the fixed box occupies so content below
        # doesn't render underneath/hidden by it.
        st.markdown('<div class="pred-sticky-spacer"></div>', unsafe_allow_html=True)

        with main_box.container():
            bag_ships = df_bag["ship_count"].sum() if not df_bag.empty else 0
            st.markdown(
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

        st.divider()

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

        table_heading("🕐 Select Cutoff")
        cutoff_options = [
            f"{r['Cutoff']} — {r['Total Shipment']:,} pending"
            for _, r in cutoff_tbl.iterrows()
        ]
        cutoff_label_to_value = dict(zip(cutoff_options, cutoff_tbl["Cutoff"]))

        with st.form("cutoff_form", border=False):
            cur_cutoff = st.session_state.sel_cutoffs[0] if st.session_state.sel_cutoffs else None
            cur_label  = next((lbl for lbl, v in cutoff_label_to_value.items() if v == cur_cutoff), None)
            chosen_label = st.selectbox(
                "Cutoff (with pending load)",
                cutoff_options,
                index=cutoff_options.index(cur_label) if cur_label in cutoff_options else 0,
            )
            submitted_cut = st.form_submit_button("✅ Apply Cutoff", use_container_width=True)

        if submitted_cut:
            new_cutoffs = [cutoff_label_to_value[chosen_label]]
            if new_cutoffs != st.session_state.sel_cutoffs:
                st.session_state.sel_dh_names = []
            st.session_state.sel_cutoffs = new_cutoffs

        sel_cutoffs = st.session_state.sel_cutoffs

        dh_loads_map = {}
        dh_summary   = pd.DataFrame()

        if not sel_cutoffs:
            table_heading("🏭 DH Load Breakdown")
            st.info("👆 Select a cutoff above and click **Apply** to see the DH breakdown.")
        else:
            filt_dh = df_dh[df_dh["cutoff_display"].isin(sel_cutoffs)].copy()
            dh_summary, dh_loads_map = build_dh_rows(filt_dh, all_dh_loads, dh_max_vehicle, vcaps)

            table_heading(f"🏭 DH Load Breakdown — {len(dh_summary)} DH(s) with pending load")

            if dh_summary.empty:
                st.success("✅ No pending floor load for any DH in the selected cutoff.")
            else:
                dh_styled = dh_summary.style.map(
                    _vehicle_badge_style, subset=["Recommended Vehicle"]
                )
                with st.form("dh_form", border=False):
                    submitted_dh = st.form_submit_button("✅ Confirm DH Selection", use_container_width=True)
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
                            "Max Vehicle Size":    st.column_config.TextColumn(alignment="center"),
                            "Recommended Vehicle": st.column_config.TextColumn(alignment="center"),
                            "Utilization %":       st.column_config.ProgressColumn(
                                format="%.1f%%", min_value=0, max_value=100,
                            ),
                            "Status":              st.column_config.TextColumn(alignment="center", width="small"),
                        },
                    )

                if submitted_dh:
                    st.session_state.sel_dh_names = [dh_summary.iloc[i]["DH Name"] for i in dh_evt.selection.rows]

        sel_names = [n for n in st.session_state.sel_dh_names if n in dh_loads_map]
        render_prediction_box(main_box, sel_names, dh_loads_map, dh_max_vehicle, vcaps, max_cap)

    # ── Tab 2: Ready to Dispatch DHs (Utilization % > 70, across all cutoffs) ──
    else:
        with st.container(key="ready_pred_sticky"):
            st.caption(f"📅 Data last updated: {last_updated.strftime('%d %b %Y, %I:%M %p')}")
            ready_main_box = st.empty()
        st.markdown('<div class="pred-sticky-spacer"></div>', unsafe_allow_html=True)

        ready_summary_all, ready_loads_map = build_dh_rows(df_dh, all_dh_loads, dh_max_vehicle, vcaps)
        ready_summary = (
            ready_summary_all[ready_summary_all["Utilization %"] > 70]
            .sort_values("Utilization %", ascending=False)
            .reset_index(drop=True)
            if not ready_summary_all.empty else ready_summary_all
        )

        table_heading(f"🚀 Ready to Dispatch DHs — {len(ready_summary)} DH(s) over 70% utilization")

        ready_sel_names = []
        if ready_summary.empty:
            st.info("No DHs currently have a recommended-vehicle utilization above 70%.")
        else:
            ready_styled = ready_summary.style.map(
                _vehicle_badge_style, subset=["Recommended Vehicle"]
            )
            with st.form("ready_dh_form", border=False):
                ready_submitted = st.form_submit_button("✅ Confirm DH Selection", use_container_width=True)
                ready_evt = st.dataframe(
                    ready_styled,
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
                        "Max Vehicle Size":    st.column_config.TextColumn(alignment="center"),
                        "Recommended Vehicle": st.column_config.TextColumn(alignment="center"),
                        "Utilization %":       st.column_config.ProgressColumn(
                            format="%.1f%%", min_value=0, max_value=100,
                        ),
                        "Status":              st.column_config.TextColumn(alignment="center", width="small"),
                    },
                )

            if ready_submitted:
                st.session_state.ready_sel_dh_names = [ready_summary.iloc[i]["DH Name"] for i in ready_evt.selection.rows]

            ready_sel_names = [n for n in st.session_state.ready_sel_dh_names if n in ready_loads_map]

        render_prediction_box(ready_main_box, ready_sel_names, ready_loads_map, dh_max_vehicle, vcaps, max_cap)


def render_about_credits():
    st.divider()
    st.markdown("ⓘ **About / Credits**")
    st.markdown(
        "Vehicle Load Prediction Dashboard  \n"
        "Version: 1.0  \n"
        "Developed by: Nitesh Kumar  \n\n"
        "*Internal tool for vehicle load planning and prediction.*"
    )


if __name__ == "__main__":
    main()
    render_about_credits()
