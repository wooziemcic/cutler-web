# site_app.py
# Streamlit app with 7 modern batch buttons to run scrapers → excerption → compile
# Keeps existing runnable batches unchanged, but displays many more fund families per batch.

import os
import random
import subprocess
from datetime import datetime

import streamlit as st
from PyPDF2 import PdfMerger


from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
import json



# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUTLER_DIR = os.path.join(SCRIPT_DIR, "Cutler")
COMPILED_DIR = os.path.join(CUTLER_DIR, "Compiled")
os.makedirs(COMPILED_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# 1) KEEP EXISTING RUNNABLE BATCHES EXACTLY AS-IS  (IDs must match 03_14_25_<Fund>.py)
# --------------------------------------------------------------------------------------
RUNNABLE_BATCHES = {
    "Batch 1": ["Amana", "Allianz", "Artisan", "Appleseed", "Buffalo", "Clarkston"],
    "Batch 2": ["Ariel", "Baron", "Baird", "Causeway", "CavanalHill", "Clipper"],
    "Batch 3": ["FirstEagleFund", "Gabelli", "Longleaf"],
    "Batch 4": ["Oakmark", "PFF", "Sequoia", "Touchstone", "Transamerica", "TRowe"],
    "Batch 5": ["Tweedy", "ValueLine", "VanEck", "Victory", "Virtus", "Wasatch"],
    "Batch 6": ["Weitz", "William", "Alger", "ACI", "ALPS", "Brookfield"],
    "Batch 7": ["AXA", "CohenSteers", "DodgeCox", "Driehaus", "Tortoise", "TRowe"],
}
RUNNABLE_FUNDS = {f for flist in RUNNABLE_BATCHES.values() for f in flist}

# --------------------------------------------------------------------------------------
# 2) BIG DISPLAY-ONLY LIST (from you). We’ll distribute these across the 7 batches.
#    Duplicates are deduped; runnable IDs remain in their original batches.
# --------------------------------------------------------------------------------------
_BIG_LIST_RAW = """
AB (AllianceBernstein)
AQR Funds
AXA Equitable
AXS Investments
Aegis Funds
Alger
Allianz Global Investors
Alps Funds
Amana Mutual Funds
American Century Investments
American Century Investments (ACI)
American Funds
Amundi US
Appleseed Fund
Ariel Investments
Ariel Investments
Artisan Partners
BNY Mellon
Baird Funds
Baron Capital
Baron Funds
BlackRock
Boston Partners
Boston Trust Walden
Brandywine Global
Brown Advisory
Buffalo Funds
Calamos Investments
Calamos Investments
Cambiar Investors
Causeway Capital
Causeway Capital Management
Cavanal Hill Funds
Centerstone Investors
Chiron Investment Management
Clarkston Capital
ClearBridge Investments
Clipper Fund
Cohen & Steers
Columbia Threadneedle
Commerce Funds
Commerce Trust
Crossmark Global
Cullen Funds
Davidson Investment
Delaware Funds
Delaware Investments
Destra Capital
Dodge & Cox
Dreyfus Corporation
Driehaus Capital Management
Eaton Vance
Eventide
Eventide Funds
FPA Funds
Federated Hermes
Federated Hermes
Fidelity Advisor Funds
Fidelity Investments
Fidelity Investments
Fidelity Investments
Fiera Capital
First Eagle
First Eagle Fund
Foundry Partners
Frank Funds
Franklin Templeton
Gabelli Funds
Gerstein Fisher
Goldman Sachs Asset Management
Grandeur Peak
Grandeur Peak Global Advisors
Guardian Capital
Guggenheim Investments
Harbor Capital
Harbor Funds
Heartland Advisors
Hennessy Funds
Homestead Funds
Integrity Viking
Invesco
J.P. Morgan
Janus Fund
Janus Henderson
John Hancock Investments
Keeley Funds
Kinetics Mutual Funds
Lazard Asset Management
Legg Mason
Lincoln Financial
LoCorr Funds
Longleaf Partners
Longleaf Partners
Longleaf Partners
Longleaf Partners
Lord Abbett
Lord Abbett
MFS Investment Management
MFS Investment Management
MFS Investment Management
MP63 Fund
MainStay Investments
MassMutual
MassMutual RetireSmart
Matthews Asia
Mesirow Financial
Morgan Stanley Investment Mgmt
Nationwide Funds
Natixis Investment Managers
Neuberger Berman
New York Life Investments
North Square Investments
Northern Trust
Nuveen
Oakmark Fund
Oakmark Fund
Oakmark Fund
Oakmark Fund
Oakmark Funds
Old Mutual Asset Management
Osterweis Capital
PGIM Investments
PIMCO
Pax World Funds
Polen Capital
Poplar Forest Funds
Principal
Putnam Investments
Queens Road Funds
Reynders McVeigh
Royce Investment Partners
Russell Investments
Scout Investments
Seafarer Capital Partners
Sequoia Funds
Shelton Capital
Smead Capital
State Street Global Advisors
Steward Partners
T. Rowe Price
T. Rowe Price
TCW Group
TIAA
Third Avenue
Thornburg Investment Mgmt
Thrivent Funds
Tocqueville Asset Management
Torray Fund
Torray Resolute
Tortoise Ecofin
Touchstone
Touchstone Investments
Touchstone Sands
Transamerica
Tweedy, Browne Company
Value Line Funds
VanEck
Victory Capital
Virtue Funds
Virtus Investment Partners
WCM Investment Mgmt
Wasatch Global Investors [Check main.py]
Weitz Investment Management
Westwood Holdings
William Blair
Yacktman Funds
Zevenbergen Capital

1290 Funds
1919 Funds
1WS Capital
AFA Funds
AGF Investments
AMG
ARK
Abbey Capital
Absolute Investment Advisers
Adirondack
Adler
AdvisorShares
Advisors Asset Management
Akre Capital
Allspring
Anchor Capital
Angel Oak
Apollo
Aptus
Archer
Ares Capital
Aristotle
Ashmore
Aspiration
Aspiriant
BBH
Baillie Gifford
Belmont Capital
Berkshire
Bexil
Blue Current
Blueprint
Boyar
Boyd Watterson
Brandes
Bretton Capital
Bridges
Bright Rock
CCT Asset Management
CRM Funds
Capital Group
Carillon
Castle Investment Management
Catholic Responsible Investments
Champlain
Chesapeake
Clark Fork
Clifford Capital
Clough Capital
Community Capital Management
Conestoga
Congress Asset Management
Copeland
Core Alternative
CornerCap
Cornerstone
Cove Street Capital
Covered Bridge
Cromwell
CrossingBridge
Cushing
Cutler Investment Group
DWS
Dana Funds
Davenport
Davis
Dean Mutual Funds
Dearborn Partners
Diamond Hill
Dinosaur Group
Direxion
Distillate Capital
Domini
Duff & Phelps
Dupree
E-Valuator
Easterly
Edgar Lomax
Edgewood
Empower
EquityCompass
FMI Funds
FS Investments
Fairholme
Fenimore Asset Management
First Pacific
Firsthand Funds
Flaherty & Crumrine
Forester Funds
Fort Pitt Capital
Frontier Funds
Frost Funds
GMO
GQG Partners
Gator Capital
Geneva
Glenmede
Global X
GoodHaven
Grayscale
GuideStone
Guinness Atkinson
Hamlin Funds
Harding Loevner
HartFord
Haverford
Hotchkis & Wiley
Hussman
IDX
IPS Strategic Capital
Impax
Innovator
Intrepid Capital
Jacob
James
Johnson
Kayne Anderson
Kirr, Marbach Partners
Knights of Columbus
Kopernik
LSV
Lawson Kroeker
Leavell
Leuthold
Long Short Advisors
Lyrical
MH Elite
Madison
MainGate
Mairs & Power
Manor Funds
Marsico
Meehan
Meridian
MetLife
Midas
Miller/Howard
Mondrian
Motley Fool
Mundoval
Mutual of America
Muzinich
Needham
NexPoint
Nicholas
North Country
NorthQuest
Northeast Investors
O'Shaughnessy
OCM
OTG
Oberweis
Old Westbury
Otter Creek
Overlay Shares
PIA
PT Asset Management
Pacer
Palm Valley
Panoramic
Papp
Paradigm
Parnassus
Peer Tree
Plan Investment Fund
Plumb
Popular Family of Funds
Potomac
Primark
Provident
Prudential
Quantified Funds
RBC
RMB
Ranger
Reaves
Recurrent
Rice Hall James
RiverPark
SBAuer
Sarofim
Selected Funds
Seven Canyons
Shenkman
Sound Shore
SouthernSun
Spinnaker ETF Trust
Sprott
Standpoint
Summit Global
Summitry
Tanaka
The Private Shares
Tributary
Trillium
U.S. Global Investors
Union Street Partners
VELA
Variant
Villere
Voya
Water Island Capital
Wilshire
Wireless
WisdomTree
Yorktown
abrdn
iMGP Funds

ACCLEVITY
ACR
ALPHCNTRC
Ave Maria Funds
BAHL & GAYNOR
Barrett
Baywood
BLACK OAK
BOSTON PART
Calvert
Centre Global
CIBC ATLAS
Clarity Fund Inc
Clifford Capital
Cohen & Steers
Column
Commonwealth Funds
CS McKee Collective Funds
Dearborn Partners
DF DENT
Diamond Hill
DOMINI IMPACT
DSM Funds
Eagle Energy
Enterprise Funds
EP EMERGING MARKETS
Europac
FAM
FIRST EAGLE
FRONTIER MFG
Galliard Funds
GAMCO
Glenmeade IM
Greenspring
Hundredfold
ICON Funds
ISHARES
Ivy
JamesAdvantage Funds
JH FINANCIAL
JP Morgan
Matrix Asset Advisors Fund
Members Funds
MG TRUST CTF
Muhlenkamp Funds
Oberweiss Funds
PAX World
Payson Funds
PrimeCap Odyssey
RidgeWorth Funds
River Park Funds
Rogue Fudns
RS Investments
Rydex Funds
TransAmerica/Idex
VILLERE FUNDS
ZACKS
"""

def _parse_display_list(raw: str) -> list[str]:
    items = [x.strip() for x in raw.splitlines()]
    items = [x for x in items if x]  # drop blanks
    # de-duplicate while preserving order
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

DISPLAY_ONLY_FUNDS = _parse_display_list(_BIG_LIST_RAW)

# --------------------------------------------------------------------------------------
# 3) BUILD DISPLAY BATCHES
#    - Start with the runnable IDs already assigned.
#    - Add extra display-only names round-robin across 7 batches.
#    - Avoid duplicates inside a batch.
# --------------------------------------------------------------------------------------
BATCH_NAMES = list(RUNNABLE_BATCHES.keys())
DISPLAY_BATCHES = {bn: list(RUNNABLE_BATCHES[bn]) for bn in BATCH_NAMES}

# extra display names that are *not* exact runnable IDs (avoid obvious duplicates)
extras = [n for n in DISPLAY_ONLY_FUNDS if n not in RUNNABLE_FUNDS]
random.seed(42)  # deterministic shuffle (looks random but repeatable)
random.shuffle(extras)

# round-robin assign extras
for idx, name in enumerate(extras):
    bn = BATCH_NAMES[idx % len(BATCH_NAMES)]
    if name not in DISPLAY_BATCHES[bn]:
        DISPLAY_BATCHES[bn].append(name)

# --------------------------------------------------------------------------------------
# Utilities (unchanged behavior for runnable funds)
# --------------------------------------------------------------------------------------
def script_for(fund: str) -> str:
    return os.path.join(SCRIPT_DIR, f"03_14_25_{fund}.py")

def run_scraper(fund: str, timeout_sec: int = 600) -> str:
    path = script_for(fund)
    if not os.path.exists(path):
        return "script_missing"
    try:
        proc = subprocess.Popen(
            ["python", path],
            cwd=SCRIPT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        try:
            ret = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            finally:
                return "script_failed"

        dld = os.path.join(CUTLER_DIR, fund, "downloads")
        got = []
        if os.path.isdir(dld):
            got = [f for f in os.listdir(dld) if f.lower().endswith(".pdf")]

        if ret == 0 and got:
            return "ok"
        elif ret == 0 and not got:
            return "no_pdfs"
        else:
            return "script_failed"
    except Exception:
        return "script_failed"

def run_excerpt(fund: str) -> bool:
    try:
        from excerpt import process_fund
        process_fund(fund)
        return True
    except Exception:
        return False
    
def _fund_folder_candidates(fid: str):
    """Yield both canonical and pretty-mapped folder names for a fund."""
    base = os.path.join(CUTLER_DIR, fid)
    yield base
    try:
        # Use the same FUND_FOLDER_MAP logic excerpt.py relies on
        from excerpt import FUND_FOLDER_MAP as _MAP  # :contentReference[oaicite:4]{index=4}
        alt = _MAP.get(fid, fid.replace(" ", "_"))
        if alt and alt != fid:
            yield os.path.join(CUTLER_DIR, alt)
    except Exception:
        pass

def _load_meta_for_fund(fid: str):
    """
    Try to load commentary metadata for a fund:
      - primary: Cutler/<Fund>/downloads/metadata.json (if your scraper wrote it)
      - fallback: infer 'commentary_title' from commentary/merged filename
    Returns dict: {'fund_name','commentary_title','runtime_ts'}
    """
    # Friendly display name
    fund_name = fid
    try:
        from excerpt import FUND_FOLDER_MAP as _MAP  # :contentReference[oaicite:5]{index=5}
        # Use the original pretty label if available in the map’s *keys*
        for k, v in _MAP.items():
            if v == fid.replace(" ", "_") or v == fid:
                fund_name = k
                break
    except Exception:
        pass

    # Defaults
    meta = {
        "fund_name": fund_name,
        "commentary_title": None,
        "runtime_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Look for metadata.json or infer from downloads
    for base in _fund_folder_candidates(fid):
        dld = os.path.join(base, "downloads")
        if not os.path.isdir(dld):
            continue

        # 1) metadata.json
        mpath = os.path.join(dld, "metadata.json")
        if os.path.exists(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    j = json.load(f)
                meta["fund_name"] = j.get("fund_name") or meta["fund_name"]
                meta["commentary_title"] = j.get("commentary_title") or meta["commentary_title"]
                # Prefer the metadata run timestamp if provided
                meta["runtime_ts"] = j.get("runtime_ts") or meta["runtime_ts"]
                return meta
            except Exception:
                pass

        # 2) fallback: infer commentary title from the most relevant file name in downloads
        # Prefer *_Merged.pdf, else any .pdf that looks like commentary; finally any .pdf
        try:
            pdfs = [f for f in os.listdir(dld) if f.lower().endswith(".pdf")]
            merged = [f for f in pdfs if f.endswith("_Merged.pdf")]
            pick = None
            if merged:
                pick = merged[0]
            else:
                comm_like = [f for f in pdfs if ("comment" in f.lower() or "q1" in f.lower() or "q2" in f.lower() or "q3" in f.lower() or "q4" in f.lower())]
                if comm_like:
                    pick = comm_like[0]
                elif pdfs:
                    pick = pdfs[0]
            if pick and not meta["commentary_title"]:
                # Strip extension and tidy underscores/hyphens
                stem = os.path.splitext(pick)[0]
                stem = stem.replace("_", " ").replace("-", " ").strip()
                meta["commentary_title"] = stem
        except Exception:
            pass

    return meta

def _overlay_single_page(w: float, h: float, hdr_left: str, hdr_mid: str, hdr_right: str):
    """
    Build a 1-page PDF overlay with header/footer text that we can merge onto a page.
    Positions are consistent with your excerption layout.  :contentReference[oaicite:6]{index=6}
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    left = right = 0.75 * 72   # 0.75in margins to match excerption_base.py  :contentReference[oaicite:7]{index=7}
    top = bottom = 0.75 * 72

    # Header
    c.setFont("Helvetica", 8.5)
    c.setFillColor(colors.HexColor("#0a5763"))
    if hdr_left:
        c.drawString(left, h - top + 0.35 * 72, hdr_left)
    if hdr_mid:
        # truncate very long titles for safety
        mid = (hdr_mid[:95] + "…") if len(hdr_mid) > 96 else hdr_mid
        c.drawCentredString(w / 2.0, h - top + 0.35 * 72, mid)
    if hdr_right:
        c.drawRightString(w - right, h - top + 0.35 * 72, hdr_right)

    # Footer: keep page number from the source PDF; we don’t add one here to avoid overlap
    c.save()
    buf.seek(0)
    return buf

def _stamp_pdf_with_meta(src_pdf_path: str, fund_name: str, commentary_title: str, runtime_ts: str) -> str:
    """
    Return a stamped copy path for src_pdf_path.
    We read page size from the source and overlay a header with fund/title/timestamp.
    """
    try:
        reader = PdfReader(src_pdf_path)
    except Exception:
        # If we can’t read the PDF, just return the original path
        return src_pdf_path

    writer = PdfWriter()
    # Build overlay per page (size-aware)
    for page in reader.pages:
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        ov_buf = _overlay_single_page(w, h, hdr_left=fund_name, hdr_mid=commentary_title or "", hdr_right=f"Run: {runtime_ts}")
        try:
            ov = PdfReader(ov_buf).pages[0]
            # Merge overlay
            page.merge_page(ov)
        except Exception:
            # If merge fails for any reason, keep original page
            pass
        writer.add_page(page)

    stamped_path = src_pdf_path.replace(".pdf", ".stamped.tmp.pdf")
    with open(stamped_path, "wb") as f:
        writer.write(f)
    return stamped_path

def compile_batch_pdf(batch_name: str, funds: list[str]) -> str | None:
    today = datetime.today().strftime("%Y%m%d")
    out_path = os.path.join(COMPILED_DIR, f"Compiled_{batch_name.replace(' ', '')}_{today}.pdf")
    merger = PdfMerger()
    added = 0

    def folders_for(fid: str):
        # Keep your original dual-path logic  :contentReference[oaicite:8]{index=8}
        yield os.path.join(CUTLER_DIR, fid, "excerpted")
        try:
            from excerpt import FUND_FOLDER_MAP as _MAP  # :contentReference[oaicite:9]{index=9}
            alt = _MAP.get(fid, fid.replace(" ", "_"))
        except Exception:
            alt = fid.replace(" ", "_")
        if alt and alt != fid:
            yield os.path.join(CUTLER_DIR, alt, "excerpted")

    for fund in funds:
        meta = _load_meta_for_fund(fund)  # fund_name, commentary_title, runtime_ts
        for folder in folders_for(fund):
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if f.startswith("Excerpted_") and f.lower().endswith(".pdf"):
                    try:
                        src = os.path.join(folder, f)
                        # Stamp per-page header with fund + commentary + runtime
                        stamped = _stamp_pdf_with_meta(
                            src,
                            fund_name=meta["fund_name"],
                            commentary_title=meta["commentary_title"] or "",
                            runtime_ts=meta["runtime_ts"],
                        )
                        merger.append(stamped)
                        added += 1
                    except Exception:
                        pass

    if added:
        try:
            merger.write(out_path)
        finally:
            merger.close()
        return out_path
    else:
        merger.close()
        return None

# --------------------------------------------------------------------------------------
# RUNNER: unified progress bar (no counts; runs only runnable funds)
# --------------------------------------------------------------------------------------
def run_batch(batch_name: str):
    st.markdown(f"### Running {batch_name}")

    # The list shown on the card (display) and the list we actually run (runnable)
    display_names = DISPLAY_BATCHES[batch_name]
    runnable_ids = RUNNABLE_BATCHES.get(batch_name, [])

    # unified progress bar with neutral text (no numbers)
    holder = st.empty()
    prog = holder.progress(0.0, text="Working…")

    # 1) Download (0 → 50%), only for runnable IDs
    total = max(len(runnable_ids), 1)
    states = {}
    for i, fund in enumerate(runnable_ids, start=1):
        frac = i / (2 * total)
        prog.progress(frac, text="Working…")
        states[fund] = run_scraper(fund)

    # 2) Excerpt (50 → 100%)
    to_excerpt = [f for f, s in states.items() if s == "ok"]
    ex_total = max(len(to_excerpt), 1)
    for j, fund in enumerate(to_excerpt, start=1):
        frac = 0.5 + (j / (2 * ex_total))
        prog.progress(frac, text="Working…")
        run_excerpt(fund)

    # 3) Compile (snap to 100%)
    prog.progress(1.0, text="Finalizing…")
    final_pdf = compile_batch_pdf(batch_name, to_excerpt)
    holder.empty()

    if final_pdf:
        st.success(f"Compiled PDF → {final_pdf}")
    else:
        st.info("Compiled output is not available yet.")

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Mutual Fund Commentary Excerption", page_icon="🗂️", layout="wide")

# Modern styling (kept; tweaked: hide any 0% progress bars, prettier chips)
st.markdown(
    """
    <style>
      .main {padding-top: 1rem;}
      .stButton>button {
        background: linear-gradient(90deg, #0a5763 0%, #2b8c9e 100%);
        color: white; border: 0; border-radius: 12px; padding: 0.75rem 1.25rem;
        font-weight: 600; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      }
      .stButton>button:hover { filter: brightness(1.05); }
      .batch-card {
        background: rgba(10,87,99,0.06);
        border: 1px solid rgba(10,87,99,0.15);
        border-radius: 16px;
        padding: 16px;
      }
      .batch-title { font-weight:700; margin-bottom:8px; }
      .fund-chip {
        display:inline-block; margin: 6px 6px 0 0; padding: 6px 12px; border-radius: 14px;
        background: #eef6f7; color: #0a5763; font-size: 12px; font-weight: 600;
        border: 1px solid rgba(10,87,99,0.15);
        white-space: nowrap;
      }
      /* collapse stray 0% progress bars if any */
      div[role="progressbar"][aria-valuenow="0"] {
        height: 0 !important; min-height: 0 !important; margin: 0 !important; padding: 0 !important;
        background: transparent !important; border: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Mutual Fund Commentary Excerption")
st.caption("Run any of the seven batches below. Each button executes: scraper → excerption → compiled PDF.")

# Controls row: Run All
col_all, _, _ = st.columns([1, 3, 3])
with col_all:
    if st.button("Run All Batches"):
        for bname in BATCH_NAMES:
            with st.container():
                run_batch(bname)
        st.stop()

# 7 batch cards (3-column grid; no numeric counts shown)
cols = st.columns(3)
for i, batch_name in enumerate(BATCH_NAMES):
    with cols[i % 3]:
        st.markdown("<div class='batch-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='batch-title'>{batch_name}</div>", unsafe_allow_html=True)

        # show chips for *all* display names (runnable + extras)
        chips_html = "".join([f"<span class='fund-chip'>{name}</span>" for name in DISPLAY_BATCHES[batch_name]])
        st.markdown(chips_html, unsafe_allow_html=True)

        if st.button(f"Run {batch_name}", key=f"btn_{batch_name}"):
            run_batch(batch_name)

        st.markdown("</div>", unsafe_allow_html=True)
