# sa_app.py
# ------------------------------------------------------------
# Cutler — Seeking Alpha Streamlit UI
# Uses seekingalpha_excerpts.py (Selenium-based symbol news scraper)
# and sa_news_ai.py (OpenAI-based news digest)
#
# Run:
#   streamlit run sa_app.py
#

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
BASE = HERE / "BSD"
BASE.mkdir(parents=True, exist_ok=True)

# Make sure we can import local modules
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import seekingalpha_excerpts as sa_mod  # type: ignore[import]
import sa_news_ai as sa_ai  # type: ignore[import]

SAConfig = sa_mod.SAConfig
run_sa = sa_mod.run_seekingalpha_excerpts


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _last_completed_us_quarter(now: Optional[datetime] = None) -> str:
    """
    Return something like '2025 Q3' for the last fully completed US quarter.
    """
    if now is None:
        now = datetime.now()

    y = now.year
    m = now.month

    if 1 <= m <= 3:
        # we are in Q1 -> last completed is previous year's Q4
        y -= 1
        q = 4
    elif 4 <= m <= 6:
        q = 1
    elif 7 <= m <= 9:
        q = 2
    else:
        q = 3

    return f"{y} Q{q}"


def _normalise_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _manifest_to_df(items: List[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        tickers = it.get("tickers") or []
        ticker_str = ", ".join(tickers) if tickers else it.get("forced_ticker", "")
        rows.append(
            {
                "Ticker": ticker_str,
                "Title": it.get("title", ""),
                "Published": it.get("published", ""),
                "Source": it.get("source", ""),
                "Feed URL": it.get("feed_url", ""),
                "Excerpts JSON": it.get("excerpts_json", ""),
            }
        )
    return pd.DataFrame(rows)


def _load_news_table(excerpts_path: Path) -> Optional[pd.DataFrame]:
    if not excerpts_path.is_file():
        return None

    data = json.loads(excerpts_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not items:
        return pd.DataFrame()

    rows = []
    for n in items:
        rows.append(
            {
                "Date": n.get("date", ""),
                "Headline": n.get("headline", ""),
                "Source": n.get("source", ""),
                "URL": n.get("url", ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Cutler — Seeking Alpha Ingestion",
    layout="wide",
)

st.title("Cutler — Seeking Alpha Ingestion")
st.caption(
    "Pull public Seeking Alpha symbol news, store them into BSD/Excerpts, "
    "and explore the headlines per ticker. An AI layer summarises the overall tone."
)

# ----- Sidebar controls -----
with st.sidebar:
    st.header("Run settings")

    default_quarter = _last_completed_us_quarter()
    quarter = st.text_input("Quarter label", value=default_quarter)

    days = st.number_input(
        "Lookback window (days)",
        min_value=1,
        max_value=90,
        value=14,
        step=1,
    )

    min_words = st.number_input(
        "Min words per paragraph (unused for SA news, kept for compatibility)",
        min_value=0,
        max_value=200,
        value=60,
        step=10,
    )

    build_pdf = st.checkbox(
        "Also build per-article PDF (reserved for future use)", value=False
    )

    tickers_raw = st.text_input("Tickers (comma-separated)", value="TSLA")
    extra_rss_raw = st.text_area(
        "Additional RSS URLs (one per line - not used in SA symbol scraping)",
        value="",
        height=80,
    )

    run_button = st.button(
        "Run Seeking Alpha ingestion", use_container_width=True
    )

st.divider()
st.subheader("Results")

# ----- Run ingestion -----
if run_button:
    tickers = _normalise_tickers(tickers_raw)
    rss_urls = [u.strip() for u in extra_rss_raw.splitlines() if u.strip()]

    if not tickers:
        st.warning("Please provide at least one ticker (e.g. TSLA).")
    else:
        cfg = SAConfig(
            quarter=quarter,
            tickers=tickers,
            days=int(days),
            min_words=int(min_words),
            rss_urls=rss_urls or None,
            build_pdf=bool(build_pdf),
            base_dir=BASE,
        )

        with st.spinner("Running Seeking Alpha ingestion..."):
            items = run_sa(cfg)

        # Fallback: if scraping returned nothing, try loading existing manifest
        if not items:
            quarter_slug = quarter.strip().replace(" ", "_")
            manifest_path = BASE / "Manifests" / f"SeekingAlpha_{quarter_slug}.json"
            if manifest_path.is_file():
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                items = data.get("items", [])
                st.info(
                    "No new items scraped; showing items from existing manifest "
                    f"for {quarter} instead."
                )

        if not items:
            st.info("No items found for the selected window and inputs.")
        else:
            # Manifest DF
            df_manifest = _manifest_to_df(items)

            st.success(f"Completed. Tickers processed: {len(df_manifest)}")

            # Download manifest button
            manifest_payload = {
                "quarter": quarter,
                "created_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "items": items,
            }
            buf = BytesIO()
            buf.write(json.dumps(manifest_payload, indent=2).encode("utf-8"))
            buf.seek(0)

            st.download_button(
                "Download run manifest (JSON)",
                data=buf,
                file_name=f"manifest_sa_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
                use_container_width=True,
            )

            st.markdown("#### Run manifest")
            st.dataframe(df_manifest, use_container_width=True, height=250)

            # ----- Per-ticker news table -----
            st.markdown("#### Ticker news details")

            ticker_options = df_manifest["Ticker"].unique().tolist()
            selected_ticker = st.selectbox(
                "Select ticker", ticker_options, index=0
            )

            if selected_ticker:
                chosen_item = None
                for it in items:
                    tickers_field = it.get("tickers") or []
                    tick_str = ", ".join(tickers_field) if tickers_field else it.get(
                        "forced_ticker", ""
                    )
                    if tick_str == selected_ticker:
                        chosen_item = it
                        break

                if chosen_item is None:
                    st.warning("No manifest item found for that ticker.")
                else:
                    ex_path = Path(chosen_item.get("excerpts_json", ""))

                    # Raw news table
                    df_news = _load_news_table(ex_path)
                    if df_news is None:
                        st.warning(f"Excerpts file not found: {ex_path}")
                    elif df_news.empty:
                        st.info("Excerpts JSON contains no news items.")
                    else:
                        st.dataframe(
                            df_news,
                            use_container_width=True,
                            height=400,
                        )

                # ----- AI news digest -----
                st.markdown("#### AI News Digest")

                if ex_path.is_file():
                    with st.spinner(f"Calling OpenAI for {selected_ticker} news summary..."):

                        # Load items saved by seekingalpha_excerpts
                        payload = json.loads(ex_path.read_text(encoding="utf-8"))
                        news_items = payload.get("items", [])

                        # NEW correct summariser call
                        summary = sa_ai.summarise_symbol_news_as_digest(
                            news_items=news_items,
                            ticker=selected_ticker,
                            model="gpt-4o-mini"
                        )

                    # Render summary
                    stance = summary.get("stance", "neutral")
                    confidence = summary.get("confidence", 0.0)
                    reasoning = summary.get("primary_reasoning", "")
                    evidence = summary.get("news_evidence", [])
                    window = summary.get("time_window", {})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Stance", str(stance).title())
                    with col2:
                        st.metric("Confidence", f"{float(confidence):.2f}")

                    if window:
                        st.caption(f"News window: {window.get('start')} → {window.get('end')}")

                    st.write(reasoning)

                    if evidence:
                        st.markdown("**Key themes from recent news:**")
                        for bullet in evidence:
                            st.markdown(f"- {bullet}")
                else:
                    st.info("AI summary not available because excerpts JSON does not exist.")
else:
    st.info(
        "Set your inputs in the sidebar and click **Run Seeking Alpha ingestion**."
    )

st.divider()
st.write("Output root on this machine:")
st.code(str(BASE))
