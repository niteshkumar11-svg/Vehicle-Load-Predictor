"""
Vehicle Load Prediction Dashboard  —  Flipkart · Hajipur Mother Hub
• Multi-select cutoff + DH tables
• Smart vehicle recommendation targeting ~90 % utilisation
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

# ── Constants ──────────────────────────────────────────────────────────────────
SPREADSHEET_ID = "1SbLc5pt0YPDBEQVOaOfyd-AJfvhTthQ5zUAcGgFU7Tc"
GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Capacity per 32 Ft truck (business rules)
BAG_SHIPMENTS_32FT = 17_000   # total shipments (in bags) per 32 Ft
SEMI_32FT          = 1_800    # semi-large units per 32 Ft
TOTES_32FT         = 650      # totes per 32 Ft
SECONDARY_32FT     = 14_235   # regular / secondary shipments per 32 Ft
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

TARGET_UTIL = 0.90   # target utilisation

# ── Page config ────────────────────────────────────────────────────────────────
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
.kcard{background:white;border-radius:14px;padding:16px 20px;border:1px solid #e2e8f0;
       box-shadow:0 2px 8px rgba(0,0,0,.05);border-left:4px solid var(--ac)}
.klabel{font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.6px}
.kvalue{font-size:34px;font-weight:900;color:var(--ac);line-height:1.1;margin:2px 0}
.ksub  {font-size:12px;color:#94a3b8;margin-top:2px}
.predcard{background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:16px;
          padding:24px 26px;color:white;box-shadow:0 8px 28px rgba(37,99,235,.35)}
.sec-hdr{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
         color:#64748b;margin:0 0 6px}
.bartrack{background:#e2e8f0;border-radius:999px;height:11px;overflow:hidden;margin-top:3px}
.barfill{height:11px;border-radius:999px}
.vcap-row{background:white;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;
          margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
</style>
""", unsafe_allow_html=True)

# ── Auth ───────────────────────────────────────────────────────────────────────
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

# ── Helpers ────────────────────────────────────────────────────────────────────
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

# ── Parse ──────────────────────────────────────────────────────────────────────
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

    # Semi-large
    semi_v = _find(sheets, "semi")
    df_semi = pd.DataFrame()
    if semi_v:
        d = _df(semi_v); dc = _destcol(d)
        if dc:
            df_semi = d[[dc]].rename(columns={dc:"destination"})
            df_semi["destination"] = df_semi["destination"].astype(str).str.strip()
            df_semi = df_semi[df_semi["destination"].notna() & (df_semi["destination"] != "nan")]

    # Tote
    tote_v = _find(sheets, "tote")
    df_tote = pd.DataFrame()
    if tote_v:
        d = _df(tote_v); dc = _destcol(d)
        if dc:
            df_tote = d[[dc]].rename(columns={dc:"destination"})
            df_tote["destination"] = df_tote["destination"].astype(str).str.strip()
            df_tote = df_tote[df_tote["destination"].notna() & (df_tote["destination"] != "nan")]

    # Secondary
    sec_v = _find(sheets, "secondary")
    df_sec = pd.DataFrame()
    if sec_v:
        d = _df(sec_v)
        nc = next((c for c in d.columns if "nexthop" in c.lower()), None)
        if nc:
            df_sec = d[[nc]].rename(columns={nc:"destination"})
            df_sec["destination"] = df_sec["destination"].astype(str).str.strip()
            df_sec = df_sec[df_sec["destination"].notna() & (df_sec["destination"] != "nan")]

    # Load Capacity
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

    # DH Name Cut-Off Wise
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


# ── Load for a DH ──────────────────────────────────────────────────────────────
def _match(df, dh_name, nexthop=""):
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


# ── Vehicle logic ──────────────────────────────────────────────────────────────
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
    Returns (vehicle_name, capacity, utilisation_frac, trucks_needed).
    """
    max_cap  = max(c for _, c in vcaps)
    req_cap  = total_frac * max_cap

    if total_frac == 0:
        return None, 0, 0.0, 0

    if total_frac > 1:
        n = int(np.ceil(total_frac))
        # last partial truck
        rem_frac = total_frac - (n - 1)
        rem_cap  = rem_frac * max_cap
        last_v   = next((v for v, c in vcaps if c >= rem_cap), vcaps[-1][0])
        return f"32 Ft × {n-1} + {last_v}", max_cap, rem_frac, n

    for v, c in vcaps:
        if c >= req_cap:
            return v, c, req_cap / c, 1

    return vcaps[-1][0], vcaps[-1][1], req_cap / vcaps[-1][1], 1

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


# ── UI helpers ─────────────────────────────────────────────────────────────────
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


# ── App ─────────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🚛 Vehicle Predictor")
        st.caption("Flipkart · Hajipur Mother Hub")
        st.divider()
        st.markdown("""
**How to use**
1. Select one or more **cutoffs** from the table
2. Select one or more **DHs** from the filtered list
3. See combined floor load & vehicle recommendation
4. Use **Manual Selector** to explore any vehicle size
""")
        st.divider()
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("## 🚛 Vehicle Load Prediction Dashboard")
    st.markdown("*Hajipur Mother Hub — floor load analysis & vehicle recommendation*")
    st.divider()

    # Load data
    with st.spinner("Loading data from Google Sheets…"):
        try:
            raw  = load_sheets()
            _key = tuple(sorted(raw.keys()))
            df_bag, df_semi, df_tote, df_sec, vcaps, df_dh = parse(_key)
        except Exception as e:
            st.error(f"❌ Could not load sheet: {e}")
            st.stop()

    max_cap = max(c for _, c in vcaps)

    # ── Overview cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: kcard("🛍️ Total Bags on Floor",    f"{len(df_bag):,}",
                    f"{df_bag['ship_count'].sum() if not df_bag.empty else 0:,} shipments", "#f59e0b")
    with c2: kcard("📦 Semi-Large Shipments",    f"{len(df_semi):,}", "Floor pending", "#3b82f6")
    with c3: kcard("🧺 Totes on Floor",          f"{len(df_tote):,}", "Pending dispatch", "#8b5cf6")
    with c4: kcard("📋 Secondary Pending",        f"{len(df_sec):,}",  "Sorted, not bagged", "#ef4444")

    st.divider()

    # ── Step 1: Cutoff table ───────────────────────────────────────────────────
    if df_dh.empty:
        st.warning("⚠️ DH Name Cut-Off sheet not found.")
        return

    cutoff_tbl = (
        df_dh.groupby("cutoff_display")
        .agg(DH_Count=("dh_name", "nunique"))
        .reset_index()
        .rename(columns={"cutoff_display": "Cutoff", "DH_Count": "# DHs"})
        .sort_values("Cutoff")
        .reset_index(drop=True)
    )

    with st.expander("🕐 Step 1 — Select Cutoff(s)", expanded=False):
        st.caption("Click rows to select · Shift+click for multi-select")
        cut_evt = st.dataframe(
            cutoff_tbl,
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            hide_index=True,
            height=min(420, (len(cutoff_tbl) + 1) * 35 + 10),
        )
        sel_cutoffs = [cutoff_tbl.iloc[i]["Cutoff"] for i in cut_evt.selection.rows]
        if sel_cutoffs:
            st.success(f"Selected: {', '.join(sel_cutoffs)}")

    if not sel_cutoffs:
        st.info("👆 Expand Step 1 and select one or more cutoff rows to continue.")
        return

    # ── Step 2: DH table ──────────────────────────────────────────────────────
    filt_dh = df_dh[df_dh["cutoff_display"].isin(sel_cutoffs)].copy()
    dh_tbl  = (
        filt_dh[["dh_code","dh_name","cutoff_display"]]
        .drop_duplicates("dh_name")
        .rename(columns={"dh_code":"DH Code","dh_name":"DH Name","cutoff_display":"Cutoff"})
        .sort_values("DH Name")
        .reset_index(drop=True)
    )

    with st.expander(f"🏭 Step 2 — Select DH(s)  ·  {len(dh_tbl)} DHs across cutoff(s) {', '.join(sel_cutoffs)}", expanded=bool(sel_cutoffs)):
        st.caption("Click rows to select · Shift+click for multi-select")
        dh_evt = st.dataframe(
            dh_tbl,
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            hide_index=True,
            height=min(480, (len(dh_tbl) + 1) * 35 + 10),
        )
        sel_dh_rows = [dh_tbl.iloc[i] for i in dh_evt.selection.rows]
        if sel_dh_rows:
            st.success(f"Selected {len(sel_dh_rows)} DH(s): {', '.join(r['DH Name'] for r in sel_dh_rows)}")

    if not sel_dh_rows:
        st.info("👆 Expand Step 2 and select one or more DH rows to continue.")
        return

    # ── Aggregate load for selected DHs ───────────────────────────────────────
    st.divider()
    sel_names = [r["DH Name"] for r in sel_dh_rows]
    st.markdown(f"### 📦 Combined Floor Load — {len(sel_names)} DH(s) selected")
    with st.expander("Selected DHs", expanded=False):
        st.write(", ".join(sel_names))

    agg = dict(bag_count=0, bag_shipments=0, semi_count=0, tote_count=0, secondary_count=0)
    with st.spinner("Aggregating load…"):
        for row in sel_dh_rows:
            dh_n = row["DH Name"]
            dh_r = filt_dh[filt_dh["dh_name"] == dh_n]
            nx   = str(dh_r.iloc[0].get("nexthop","")) if "nexthop" in dh_r.columns and len(dh_r) else ""
            ld   = dh_load(dh_n, nx, df_bag, df_semi, df_tote, df_sec)
            for k in agg:
                agg[k] += ld[k]

    total_ship = agg["bag_shipments"] + agg["semi_count"] + agg["tote_count"] + agg["secondary_count"]

    m1, m2, m3, m4 = st.columns(4)
    with m1: kcard("🛍️ Bags",         f"{agg['bag_count']:,}",    f"{agg['bag_shipments']:,} shipments inside", "#f59e0b")
    with m2: kcard("📦 Semi-Large",   f"{agg['semi_count']:,}",   "Shipments on floor", "#3b82f6")
    with m3: kcard("🧺 Totes",        f"{agg['tote_count']:,}",   "Totes on floor", "#8b5cf6")
    with m4: kcard("📋 Secondary",    f"{agg['secondary_count']:,}", "Sorted, pending bag", "#ef4444")

    if total_ship == 0:
        st.success("✅ No pending floor load for the selected DHs.")
        return

    st.markdown(f"<div style='font-size:14px;color:#475569;margin:10px 0'>"
                f"Total equivalent shipments: <b style='font-size:22px;color:#1e293b'>{total_ship:,}</b></div>",
                unsafe_allow_html=True)

    # ── Pie + Vehicle recommendation ──────────────────────────────────────────
    st.divider()
    col_pie, col_rec = st.columns([1, 1])

    total_frac = load_to_frac(agg)
    req_equiv  = frac_to_equiv(total_frac, max_cap)

    with col_pie:
        labels = ["Bag Shipments", "Semi-Large", "Totes", "Secondary Pending"]
        vals   = [agg["bag_shipments"], agg["semi_count"], agg["tote_count"], agg["secondary_count"]]
        colors = ["#f59e0b","#3b82f6","#8b5cf6","#ef4444"]
        nz = [(l,v,c) for l,v,c in zip(labels,vals,colors) if v>0]
        fig = go.Figure(go.Pie(
            labels=[x[0] for x in nz], values=[x[1] for x in nz],
            marker_colors=[x[2] for x in nz], hole=0.55,
            textinfo="label+percent", textfont_size=12,
        ))
        fig.update_layout(
            showlegend=False, height=300,
            margin=dict(t=10,b=10,l=10,r=10),
            annotations=[dict(text=f"<b>{total_ship:,}</b><br>Total",
                              x=0.5,y=0.5,font_size=14,showarrow=False)],
        )
        st.markdown("**Load Bifurcation**")
        st.plotly_chart(fig, use_container_width=True)

    with col_rec:
        best_v, best_cap, best_util, n_trucks = recommend_vehicle(total_frac, vcaps)

        if best_v is None:
            st.warning("No load to predict.")
        else:
            util_pct  = round(best_util * 100, 1)
            conf_col  = "#16a34a" if util_pct >= 75 else "#f59e0b" if util_pct >= 40 else "#ef4444"

            st.markdown(
                f'<div class="predcard">'
                f'<div style="font-size:13px;opacity:.8;font-weight:500">🎯 Recommended Vehicle</div>'
                f'<div style="font-size:44px;font-weight:900;margin:6px 0 4px;letter-spacing:-1px">{best_v}</div>'
                f'<div style="display:flex;gap:24px;margin-top:10px">'
                f'<div><div style="font-size:12px;opacity:.75">Load Utilization</div>'
                f'<div style="font-size:26px;font-weight:800;color:{conf_col}">{util_pct}%</div></div>'
                f'<div><div style="font-size:12px;opacity:.75">Trucks Needed</div>'
                f'<div style="font-size:26px;font-weight:800">{n_trucks}</div></div>'
                f'<div><div style="font-size:12px;opacity:.75">Total Shipments</div>'
                f'<div style="font-size:26px;font-weight:800">{total_ship:,}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Remaining to reach 90 %
            if n_trucks == 1 and isinstance(best_cap, int):
                rem_eq = remaining_to_target(req_equiv, best_cap, max_cap)
                rem_bd = breakdown_remaining(rem_eq, max_cap)
                target_eq = int(best_cap * TARGET_UTIL)

                st.markdown("<br>", unsafe_allow_html=True)
                if rem_eq > 0:
                    st.markdown(
                        f"<div style='background:#fefce8;border:1px solid #fde047;border-radius:10px;"
                        f"padding:12px 16px;font-size:13px'>"
                        f"<b>📈 To reach 90% utilization ({target_eq:,} shipments):</b><br>"
                        f"Can accommodate <b>{int(rem_eq):,}</b> more equivalent shipments, e.g.:<br>"
                        f"&nbsp;&nbsp;• <b>{rem_bd['bags']:,}</b> more bags &nbsp;|&nbsp; "
                        f"<b>{rem_bd['semi']:,}</b> more semi-large &nbsp;|&nbsp; "
                        f"<b>{rem_bd['totes']:,}</b> more totes &nbsp;|&nbsp; "
                        f"<b>{rem_bd['secondary']:,}</b> secondary"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success(f"✅ Vehicle is at ≥90% utilization — optimal load!")

    # ── Manual Vehicle Selector ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔧 Manual Vehicle Selector")
    st.caption("Choose any vehicle to see how much it can carry and what will remain on floor.")

    vnames = [v for v, _ in vcaps]
    sel_v  = st.selectbox("Select Vehicle Size", vnames,
                           index=next((i for i,(v,_) in enumerate(vcaps) if v==best_v), 0)
                           if best_v and best_v in vnames else 0)

    sel_cap = next(c for v, c in vcaps if v == sel_v)
    sel_frac = sel_cap / max_cap               # fraction of 32 Ft
    load_eq  = req_equiv                        # equivalent shipments needed

    # What fits in this vehicle
    can_fit_eq  = min(load_eq, sel_cap)         # equiv shipments that fit
    cant_fit_eq = max(0, load_eq - sel_cap)     # equiv that won't fit

    can_fit_frac  = can_fit_eq  / max_cap
    cant_fit_frac = cant_fit_eq / max_cap

    util_sel = round(can_fit_eq / sel_cap * 100, 1) if sel_cap else 0
    util_col  = "#16a34a" if util_sel >= 80 else "#f59e0b" if util_sel >= 50 else "#ef4444"

    rem_to_90_eq = remaining_to_target(can_fit_eq, sel_cap, max_cap)
    rem_to_90_bd = breakdown_remaining(rem_to_90_eq, max_cap)

    # What can be loaded (proportional breakdown)
    can_bags_ships = int(can_fit_frac * BAG_SHIPMENTS_32FT)
    can_bags       = can_bags_ships // SHIPMENTS_PER_BAG
    can_semi       = int(can_fit_frac * SEMI_32FT)
    can_totes      = int(can_fit_frac * TOTES_32FT)
    can_sec        = int(can_fit_frac * SECONDARY_32FT)

    # What stays on floor (proportional)
    rem_bags_ships = int(cant_fit_frac * BAG_SHIPMENTS_32FT)
    rem_bags       = rem_bags_ships // SHIPMENTS_PER_BAG
    rem_semi       = int(cant_fit_frac * SEMI_32FT)
    rem_totes      = int(cant_fit_frac * TOTES_32FT)
    rem_sec        = int(cant_fit_frac * SECONDARY_32FT)

    r1, r2 = st.columns(2)
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
            f'Loading <b>{int(can_fit_eq):,}</b> of <b>{int(load_eq):,}</b> equivalent shipments</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px">'
            f'<div class="klabel" style="color:#16a34a">✅ Fits in {sel_v}</div>'
            f'<div style="font-size:13px;margin:8px 0 12px">'
            f'&nbsp;🛍️ <b>{can_bags:,}</b> bags &nbsp;({can_bags_ships:,} shipments)<br>'
            f'&nbsp;📦 <b>{can_semi:,}</b> semi-large shipments<br>'
            f'&nbsp;🧺 <b>{can_totes:,}</b> totes<br>'
            f'&nbsp;📋 <b>{can_sec:,}</b> secondary shipments'
            f'</div>'
            f'<hr style="border:none;border-top:1px solid #f1f5f9;margin:0 0 10px">'
            f'<div class="klabel" style="color:#ef4444">⏳ Remains on Floor</div>'
            f'<div style="font-size:13px;margin:8px 0">'
            f'&nbsp;🛍️ <b>{rem_bags:,}</b> bags &nbsp;({rem_bags_ships:,} shipments)<br>'
            f'📦 <b>{rem_semi:,}</b> semi-large &nbsp;|&nbsp; '
            f'🧺 <b>{rem_totes:,}</b> totes &nbsp;|&nbsp; '
            f'📋 <b>{rem_sec:,}</b> secondary'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Remaining to 90 % for selected vehicle
    if rem_to_90_eq > 0:
        st.markdown(
            f"<div style='background:#fefce8;border:1px solid #fde047;border-radius:10px;"
            f"padding:12px 16px;font-size:13px;margin-top:8px'>"
            f"<b>📈 To reach 90% in {sel_v} ({int(sel_cap*TARGET_UTIL):,} shipments):</b>&nbsp; "
            f"Can load <b>{int(rem_to_90_eq):,}</b> more equivalent shipments — e.g. "
            f"<b>{rem_to_90_bd['bags']:,}</b> bags &nbsp;|&nbsp; "
            f"<b>{rem_to_90_bd['semi']:,}</b> semi-large &nbsp;|&nbsp; "
            f"<b>{rem_to_90_bd['totes']:,}</b> totes"
            f"</div>",
            unsafe_allow_html=True,
        )

    if total_frac > 1:
        st.warning(f"⚠️ Load exceeds 1 truck — minimum **{int(np.ceil(total_frac))} vehicles** required.")

    # ── All vehicles comparison table ──────────────────────────────────────────
    with st.expander("📋 All Vehicles — Comparison Table", expanded=False):
        rows = []
        for v, c in vcaps:
            f    = min(load_eq, c)
            u    = round(f / c * 100, 1) if c else 0
            left = max(0, load_eq - c)
            r90  = max(0, c * TARGET_UTIL - f)
            rows.append({
                "Vehicle": v,
                "Capacity": f"{c:,}",
                "Can Load (equiv.)": f"{int(f):,}",
                "Utilization": f"{u}%",
                "Remains on Floor (equiv.)": f"{int(left):,}",
                "More to reach 90% (equiv.)": f"{int(r90):,}" if r90 > 0 else "✅ Optimal",
                "Trucks Needed": int(np.ceil(load_eq / c)) if c else 1,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
