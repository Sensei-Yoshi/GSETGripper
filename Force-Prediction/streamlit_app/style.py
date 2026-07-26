"""Page-wide styling for the Streamlit research lab."""

from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
:root { --ink:#17211b; --muted:#66736b; --line:#dce4df; --accent:#147a4a; --warm:#b45f19; }
.stApp { background:#f7faf8; color:var(--ink); }
.block-container { padding-top:1.4rem; max-width:1500px; }
h1, h2, h3 { letter-spacing:0 !important; }
h1 { font-size:2rem !important; }
h2 { font-size:1.35rem !important; }
[data-testid="stMetric"] { border-top:2px solid var(--line); padding-top:.7rem; }
[data-testid="stMetricValue"] { font-size:1.55rem; }
.synthetic-note { border-left:4px solid var(--warm); padding:.55rem .8rem; background:#fff9f3; }
.formula { border-left:4px solid var(--accent); padding:.45rem .8rem; background:#f0f7f3; }
.status-ok { color:var(--accent); font-weight:650; }
.status-warn { color:var(--warm); font-weight:650; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); }
button { border-radius:6px !important; }
</style>
"""


def apply_style() -> None:
    """Apply the original application CSS without changing its contents."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
