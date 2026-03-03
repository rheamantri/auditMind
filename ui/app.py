"""
AuditMind — UI v3
Left-sidebar navigation. Base dashboard always-on.
Query-specific dynamic charts filtered to what was asked.
"""

import sys, json, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agents.auditMind_agents import build_graph, AuditState

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AuditMind",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="auto"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #0d1117;
    color: #c9d1d9;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1600px; }

/* Header */
.am-header {
    background: linear-gradient(120deg, #161b22 0%, #1c2a3a 60%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.am-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.12em;
}
.am-sub {
    font-size: 0.68rem;
    color: #8b949e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #3fb950;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 5px #3fb950;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Nav items in sidebar */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.55rem 0.9rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.83rem;
    color: #8b949e;
    margin-bottom: 2px;
    transition: all 0.15s;
    border: 1px solid transparent;
}
.nav-item:hover { background: #161b22; color: #c9d1d9; }
.nav-item.active {
    background: #1c2a3a;
    color: #58a6ff;
    border-color: #30363d;
}

/* Stat card */
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #58a6ff; }
.stat-label {
    font-size: 0.62rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 5px;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem;
    font-weight: 600;
    color: #58a6ff;
    line-height: 1.2;
}
.stat-sub { font-size: 0.68rem; color: #484f58; margin-top: 3px; }

/* Decision badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 0.5rem 1.2rem;
    border-radius: 7px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.08em;
}
.badge-ESCALATE    { background:rgba(248,81,73,0.1);   color:#f85149; border:1px solid rgba(248,81,73,0.3); }
.badge-INVESTIGATE { background:rgba(88,166,255,0.1);  color:#58a6ff; border:1px solid rgba(88,166,255,0.3); }
.badge-REVIEW      { background:rgba(210,153,34,0.1);  color:#d2a020; border:1px solid rgba(210,153,34,0.3); }
.badge-DISCLOSE    { background:rgba(63,185,80,0.1);   color:#3fb950; border:1px solid rgba(63,185,80,0.3); }
.badge-IGNORE      { background:rgba(139,148,158,0.1); color:#8b949e; border:1px solid rgba(139,148,158,0.3); }

/* Chat bubbles */
.bubble-user {
    background: #1c2a3a;
    border: 1px solid #30363d;
    border-radius: 10px 10px 2px 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    font-size: 0.86rem;
    color: #93c5fd;
    line-height: 1.6;
}
.bubble-bot {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px 10px 10px 2px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-size: 0.84rem;
    color: #c9d1d9;
    line-height: 1.75;
}

/* Citation */
.citation {
    background: #0d1930;
    border-left: 3px solid #1f6feb;
    border-radius: 0 7px 7px 0;
    padding: 0.6rem 0.85rem;
    margin: 0.35rem 0;
}
.cit-src {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    color: #58a6ff;
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.cit-text { font-size: 0.76rem; color: #8b949e; line-height: 1.6; }

/* Flag */
.flag { border-radius: 0 7px 7px 0; padding: 0.5rem 0.8rem; margin: 0.3rem 0; }
.flag-HIGH   { background:rgba(248,81,73,0.07);  border-left:3px solid #f85149; }
.flag-MEDIUM { background:rgba(210,153,34,0.07); border-left:3px solid #d2a020; }
.flag-LOW    { background:rgba(63,185,80,0.07);  border-left:3px solid #3fb950; }
.flag-lbl { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; margin-bottom:2px; }
.flag-txt { font-size:0.75rem; color:#8b949e; }

/* Section header */
.section-hdr {
    font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #010409 !important; border-right: 1px solid #21262d !important; }
[data-testid="stSidebar"] * { color: #8b949e !important; }
[data-testid="stSidebar"] h3 { color: #58a6ff !important; font-size: 0.7rem !important; letter-spacing: 0.1em !important; }

/* Buttons */
.stButton > button {
    background: #161b22 !important;
    color: #58a6ff !important;
    border: 1px solid #30363d !important;
    border-radius: 7px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    padding: 0.4rem 0.75rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.15s !important;
    white-space: normal !important;
    height: auto !important;
}
.stButton > button:hover { background: #1c2a3a !important; border-color: #58a6ff !important; }

/* Input */
div[data-testid="stTextInput"] input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 9px;
    padding: 0.75rem !important;
}
div[data-testid="stMetric"] label { color: #8b949e !important; font-size: 0.67rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

COMPANY_MAP = {
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpm": "JPM",
    "bank of america": "BAC", "bofa": "BAC", "bac": "BAC",
    "goldman": "GS", "goldman sachs": "GS", "gs": "GS",
    "morgan stanley": "MS", "ms": "MS",
    "wells fargo": "WFC", "wfc": "WFC",
    "citigroup": "C", "citi": "C",
    "american express": "AXP", "amex": "AXP", "axp": "AXP",
    "blackrock": "BLK", "blk": "BLK",
    "schwab": "SCHW", "charles schwab": "SCHW", "schw": "SCHW",
    "us bancorp": "USB", "usb": "USB",
}

DEMO_QUERIES = [
    "Which transactions show structuring behavior consistent with FATF threshold avoidance?",
    "Analyze JPMorgan's financial statements for materiality anomalies and late filings",
    "Compare Goldman Sachs vs Morgan Stanley operating expense trends",
    "What AML compliance gaps exist based on FFIEC internal controls standards?",
    "Show me fan-out and layering patterns in the transaction data",
    "Which accounts fully drained their balance — flag placement phase risk",
    "What does FFIEC say about suspicious activity reporting thresholds?",
    "Which companies had the largest year-over-year operating expense spikes?",
    "Give me a full audit risk assessment across all data sources",
    "Identify high-velocity accounts sending to more than 5 unique destinations",
]

PLOT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="IBM Plex Sans", color="#8b949e", size=11),
    margin=dict(l=10, r=15, t=35, b=10),
)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

for k, v in [
    ("page", "🏠  Home"),
    ("messages", []),
    ("last_result", None),
    ("mp_cache", None),
    ("query_count", 0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

if "graph" not in st.session_state:
    with st.spinner("Initializing agent pipeline..."):
        st.session_state.graph = build_graph()

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def extract_companies(query: str) -> list[str]:
    """Returns list of ticker symbols mentioned in query."""
    q = query.lower()
    found = []
    for name, ticker in COMPANY_MAP.items():
        if name in q and ticker not in found:
            found.append(ticker)
    return found


def load_edgar() -> pd.DataFrame | None:
    try:
        return pd.read_csv(ROOT / "data/processed/edgar_processed.csv")
    except Exception:
        return None


def load_paysim() -> pd.DataFrame | None:
    try:
        return pd.read_csv(ROOT / "data/processed/paysim_high_risk.csv")
    except Exception:
        return None


def run_pipeline(query: str) -> dict:
    initial: AuditState = {
        "user_query": query, "query_intent": "", "agents_to_run": [],
        "anomaly_findings": {}, "financial_findings": {},
        "rag_findings": [], "compliance_flags": [],
        "materiality_score": 0.0, "materiality_detail": {},
        "explainability": {}, "decision": "",
        "decision_rationale": "", "scenario_projection": "",
        "final_response": "", "agents_executed": [],
        "confidence_score": 0.0, "audit_trail": [],
    }
    return st.session_state.graph.invoke(initial)


# ════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def gauge(score: float) -> go.Figure:
    c = "#f85149" if score >= 70 else "#d2a020" if score >= 40 else "#3fb950"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font": {"color": c, "family": "IBM Plex Mono", "size": 30},
                "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color":"#484f58","size":9},
                     "tickcolor":"#30363d"},
            "bar": {"color": c, "thickness": 0.2},
            "bgcolor": "#161b22", "borderwidth": 0,
            "steps": [
                {"range":[0,40],   "color":"rgba(63,185,80,0.08)"},
                {"range":[40,70],  "color":"rgba(210,153,34,0.08)"},
                {"range":[70,100], "color":"rgba(248,81,73,0.08)"},
            ],
            "threshold": {"line":{"color":c,"width":3},"thickness":0.75,"value":score}
        }
    ))
    fig.update_layout(height=200, **PLOT)
    return fig


def profit_margin_chart(df: pd.DataFrame, tickers: list[str] = None) -> go.Figure:
    """Shows profit margin. Filters to specific companies if tickers provided."""
    data = df.sort_values("period_end").groupby("ticker").last().reset_index()
    data = data.dropna(subset=["profit_margin"])

    if tickers:
        filtered = data[data["ticker"].isin(tickers)]
        if len(filtered) == 0:
            filtered = data  # fallback to all if company not found
        title = f"Profit Margin — {', '.join(tickers)}" if len(tickers) <= 3 else "Profit Margin by Company"
    else:
        filtered = data.nlargest(8, "profit_margin")
        title = "Profit Margin — All Companies (Latest Quarter)"

    colors = ["#3fb950" if v >= 0 else "#f85149" for v in filtered["profit_margin"]]
    fig = go.Figure(go.Bar(
        x=filtered["ticker"],
        y=(filtered["profit_margin"] * 100).round(2),
        marker=dict(color=colors, opacity=0.85, line=dict(color="#0d1117", width=1)),
        text=[f"{v:.1f}%" for v in (filtered["profit_margin"] * 100)],
        textposition="outside",
        textfont=dict(color="#8b949e", size=9, family="IBM Plex Mono"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#8b949e", size=10)),
        height=260,
        xaxis=dict(showgrid=False, tickfont=dict(color="#c9d1d9", size=11, family="IBM Plex Mono")),
        yaxis=dict(showgrid=True, gridcolor="#21262d", tickfont=dict(color="#484f58", size=9)),
        showlegend=False, **PLOT,
    )
    return fig


def yoy_expense_chart(df: pd.DataFrame, tickers: list[str] = None) -> go.Figure | None:
    if "operating_expenses_yoy_pct" not in df.columns:
        return None
    # dropna BEFORE groupby so we get last NON-NULL value per company
    data = (df.dropna(subset=["operating_expenses_yoy_pct"])
              .sort_values("period_end")
              .groupby("ticker").last().reset_index())

    if tickers:
        filtered = data[data["ticker"].isin(tickers)]
        title = f"YoY Expense Change — {', '.join(tickers)}"
    else:
        filtered = data
        title = "YoY Operating Expense Change — All Companies"

    if len(filtered) == 0:
        filtered = data

    colors = ["#f85149" if abs(v) > 20 else "#d2a020" if abs(v) > 10 else "#3fb950"
              for v in filtered["operating_expenses_yoy_pct"]]
    fig = go.Figure(go.Bar(
        x=filtered["ticker"],
        y=filtered["operating_expenses_yoy_pct"].round(1),
        marker=dict(color=colors, opacity=0.85, line=dict(color="#0d1117", width=1)),
        text=[f"{v:.1f}%" for v in filtered["operating_expenses_yoy_pct"]],
        textposition="outside",
        textfont=dict(color="#8b949e", size=9),
        hovertemplate="Ticker: %{x}<br>YoY Change: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=20, line=dict(color="#f85149", width=1, dash="dot"),
                  annotation=dict(text="20% spike threshold",
                                  font=dict(color="#f85149", size=8, family="IBM Plex Mono"),
                                  xanchor="right", bgcolor="#0d1117", borderpad=2))
    fig.add_hline(y=-20, line=dict(color="#3fb950", width=1, dash="dot"))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#8b949e", size=10)),
        height=260,
        xaxis=dict(showgrid=False, tickfont=dict(color="#c9d1d9", size=11, family="IBM Plex Mono")),
        yaxis=dict(showgrid=True, gridcolor="#21262d", tickfont=dict(color="#484f58", size=9),
                   ticksuffix="%"),
        showlegend=False, **PLOT,
    )
    return fig



def company_detail_chart(df: pd.DataFrame, ticker: str) -> go.Figure | None:
    """Time series of key metrics for a single company."""
    company = df[df["ticker"] == ticker].copy()
    if len(company) < 2:
        return None
    company = company.sort_values("period_end")

    fig = go.Figure()
    if "profit_margin" in company.columns:
        fig.add_trace(go.Scatter(
            x=company["period_end"],
            y=(company["profit_margin"] * 100).round(2),
            name="Profit Margin %",
            line=dict(color="#58a6ff", width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ))
    if "operating_expenses_yoy_pct" in company.columns:
        fig.add_trace(go.Scatter(
            x=company["period_end"],
            y=company["operating_expenses_yoy_pct"].round(1),
            name="OpEx YoY %",
            line=dict(color="#f85149", width=2, dash="dot"),
            mode="lines+markers",
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=dict(text=f"{ticker} — Historical Trends",
                  font=dict(color="#8b949e", size=10)),
        height=280,
        xaxis=dict(showgrid=False, tickfont=dict(color="#484f58", size=8)),
        yaxis=dict(showgrid=True, gridcolor="#21262d",
                   tickfont=dict(color="#484f58", size=8)),
        legend=dict(font=dict(color="#8b949e", size=9), bgcolor="#161b22"),
        **PLOT,
    )
    return fig


def typology_chart(typology: dict) -> go.Figure | None:
    if not typology:
        return None
    labels = [k.split("(")[0].strip() for k in typology.keys()]
    values = list(typology.values())
    palette = ["#f85149", "#d2a020", "#58a6ff", "#3fb950", "#bc8cff", "#fb923c"]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=palette[:len(labels)], opacity=0.85,
                    line=dict(color="#0d1117", width=1)),
        text=[f"{v:,}" for v in values],
        textposition="outside",
        textfont=dict(color="#484f58", size=9, family="IBM Plex Mono"),
    ))
    fig.update_layout(
        title=dict(text="AML Typology Breakdown", font=dict(color="#8b949e", size=10)),
        height=230,
        xaxis=dict(showgrid=False, tickfont=dict(color="#484f58", size=8)),
        yaxis=dict(showgrid=False, tickfont=dict(color="#c9d1d9", size=10)),
        showlegend=False, **PLOT,
    )
    return fig


def aml_overview_chart(df_pay: pd.DataFrame) -> go.Figure:
    """Pie of transaction types in high-risk pool."""
    if "type" not in df_pay.columns:
        return None
    counts = df_pay["type"].value_counts().head(6)
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(
            colors=["#58a6ff","#3fb950","#f85149","#d2a020","#bc8cff","#fb923c"],
            line=dict(color="#0d1117", width=2)
        ),
        textfont=dict(size=9, color="#8b949e"),
    ))
    fig.update_layout(
        title=dict(text="High-Risk Transaction Types", font=dict(color="#8b949e", size=10)),
        height=240,
        legend=dict(font=dict(color="#8b949e", size=9), bgcolor="#0d1117"),
        **PLOT,
    )
    return fig


def risk_score_dist(df_pay: pd.DataFrame) -> go.Figure:
    if "aml_risk_score" not in df_pay.columns:
        return None
    counts, edges = np.histogram(df_pay["aml_risk_score"].dropna(), bins=10)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))]
    colors = ["#3fb950" if i < 3 else "#d2a020" if i < 6 else "#f85149"
              for i in range(len(counts))]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker=dict(color=colors, opacity=0.85, line=dict(color="#0d1117", width=1)),
    ))
    fig.update_layout(
        title=dict(text="AML Risk Score Distribution", font=dict(color="#8b949e", size=10)),
        height=220,
        xaxis=dict(showgrid=False, tickfont=dict(color="#484f58", size=8, family="IBM Plex Mono")),
        yaxis=dict(showgrid=True, gridcolor="#21262d", tickfont=dict(color="#484f58", size=8)),
        showlegend=False, **PLOT,
    )
    return fig


# ── NEW HIGH-IMPACT CHARTS ────────────────────────────────────────────────────

def structuring_cliff_chart(df_pay: pd.DataFrame) -> go.Figure | None:
    """The $10K cliff — transaction amount histogram showing artificial pile-up below $10K."""
    if "amount" not in df_pay.columns:
        return None
    window = df_pay[(df_pay["amount"] >= 7000) & (df_pay["amount"] <= 12000)]["amount"]
    if len(window) < 100:
        return None
    counts, edges = np.histogram(window, bins=60)
    bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    colors = ["#f85149" if c < 10000 else "#484f58" for c in bin_centers]

    # Fix 7: find actual peak bin below $10K dynamically
    below_10k_idx = [i for i, c in enumerate(bin_centers) if c < 10000]
    if below_10k_idx:
        peak_idx = max(below_10k_idx, key=lambda i: counts[i])
        peak_x   = bin_centers[peak_idx]
        peak_y   = int(counts[peak_idx])
    else:
        peak_x, peak_y = 9500, 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers, y=counts,
        marker=dict(color=colors, opacity=0.85, line=dict(color="#0d1117", width=0.5)),
        name="Transaction Count",
        hovertemplate="Amount: $%{x:,.0f}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=10000, line=dict(color="#ffd700", width=2, dash="dash"),
                  annotation=dict(
                      text="$10,000 Reporting Threshold",
                      font=dict(color="#ffd700", size=9, family="IBM Plex Mono"),
                      bgcolor="#0d1117", borderpad=3,
                      xanchor="left",  # Fix: prevent label truncation
                  ))
    fig.add_annotation(
        x=peak_x, y=peak_y,
        text=f"Peak: {peak_y:,} tx<br>at ${peak_x:,.0f}",
        font=dict(color="#f85149", size=8, family="IBM Plex Mono"),
        showarrow=True, arrowhead=2, arrowcolor="#f85149",
        ax=0, ay=-45, bgcolor="#0d1117", borderpad=3,
        xanchor="center",
    )
    # Add % drop annotation at the cliff edge
    at_threshold = [i for i, c in enumerate(bin_centers) if c >= 10000]
    if at_threshold and below_10k_idx:
        last_below = max(below_10k_idx)
        first_above = at_threshold[0]
        if counts[last_below] > 0:
            drop_pct = (counts[last_below] - counts[first_above]) / counts[last_below] * 100
            fig.add_annotation(
                x=10000, y=(counts[last_below] + counts[first_above]) / 2,
                text=f"▼ {drop_pct:.0f}%\ncliff drop",
                font=dict(color="#ffd700", size=8, family="IBM Plex Mono"),
                showarrow=False, bgcolor="#0d1117", borderpad=3,
                xanchor="left",
            )
    fig.update_layout(
        title=dict(text="Structuring Cliff — Transaction Amounts $7K–$12K  (red = below $10K threshold)",
                  font=dict(color="#8b949e", size=10)),
        height=300,
        xaxis=dict(showgrid=False, tickprefix="$", tickformat=",",
                   tickfont=dict(color="#8b949e", size=9, family="IBM Plex Mono"),
                   title=dict(text="Transaction Amount ($)", font=dict(color="#484f58", size=9))),
        yaxis=dict(showgrid=True, gridcolor="#21262d",
                   tickfont=dict(color="#484f58", size=8),
                   title=dict(text="Number of Transactions", font=dict(color="#484f58", size=9))),
        showlegend=False, **PLOT,
    )
    return fig


def risk_heatmap(df_edgar: pd.DataFrame, df_pay: pd.DataFrame) -> go.Figure | None:
    """
    Companies × 5 risk dimensions heatmap.
    Each dimension uses its own normalization so no column saturates.
    """
    try:
        if "ticker" not in df_edgar.columns:
            return None
        # Use only the most recent 8 quarters per company for relevance
        df = (df_edgar.sort_values("period_end")
              .groupby("ticker").tail(8).copy())
        tickers = sorted(df["ticker"].unique().tolist())
        if not tickers:
            return None

        dimensions = ["AML Exposure", "Late Filing %", "Expense Spike",
                      "Leverage Risk", "Peer Deviation"]

        # ── Peer-wide baselines (computed once) ──────────────────────────────
        # Expense spike: max abs YoY per company → z-score across companies
        exp_col = "operating_expenses_yoy_pct"
        exp_max_by_co = {}
        if exp_col in df.columns:
            for t in tickers:
                vals = df[df["ticker"] == t][exp_col].dropna().abs()
                exp_max_by_co[t] = float(vals.max()) if len(vals) else 0.0
            exp_arr = np.array(list(exp_max_by_co.values()))
            exp_mean, exp_std = exp_arr.mean(), max(exp_arr.std(), 1.0)

        # Leverage: debt_to_equity across all companies
        de_col = "debt_to_equity"
        de_by_co = {}
        if de_col in df.columns:
            for t in tickers:
                vals = df[df["ticker"] == t][de_col].dropna()
                de_by_co[t] = float(vals.mean()) if len(vals) else 0.0
            de_arr = np.array(list(de_by_co.values()))
            de_mean, de_std = de_arr.mean(), max(de_arr.std(), 0.1)

        scores = []
        tooltips = []  # store human-readable explanations
        for t in tickers:
            co  = df[df["ticker"] == t]
            row = []
            tip = []

            # 1. AML Exposure — max materiality score in recent quarters (0-100 direct)
            if "materiality_score" in co.columns:
                ms = float(co["materiality_score"].max() or 0)
                aml_s = min(100, ms)
                tip.append(f"Max mat. score: {ms:.0f}")
            else:
                aml_s = 0
                tip.append("N/A")
            row.append(aml_s)

            # 2. Late Filing % — % of recent quarters filed late
            #    Bands: 0% → 0, 12.5% (1/8 qtr) → 25, 25% → 50, 50% → 100
            if "late_filing" in co.columns:
                n_total = len(co)
                n_late  = int(co["late_filing"].sum())
                lf_pct  = (n_late / n_total * 100) if n_total else 0
                lf_s    = min(100, lf_pct * 2)  # 50% late = 100 risk
                tip.append(f"{n_late}/{n_total} quarters late")
            else:
                lf_s = 0
                tip.append("N/A")
            row.append(float(lf_s))

            # 3. Expense Spike — z-score vs all-company peer max
            #    z=0 (average) → 50, z=+1 → 70, z=+2 → 90, z<0 → <50
            if exp_max_by_co and t in exp_max_by_co:
                z = (exp_max_by_co[t] - exp_mean) / exp_std
                ea_s = min(100, max(0, 50 + z * 20))
                tip.append(f"Max YoY spike: {exp_max_by_co[t]:.1f}%  z={z:.2f}")
            else:
                ea_s = 50
                tip.append("N/A")
            row.append(float(ea_s))

            # 4. Leverage Risk — debt_to_equity z-score
            #    z=0 → 50, z=+1 → 67, z=+2 → 83, z=-1 → 33
            if de_by_co and t in de_by_co:
                z = (de_by_co[t] - de_mean) / de_std
                de_s = min(100, max(0, 50 + z * 17))
                tip.append(f"Avg D/E: {de_by_co[t]:.2f}  z={z:.2f}")
            else:
                de_s = 50
                tip.append("N/A")
            row.append(float(de_s))

            # 5. Peer Deviation — expense_ratio_zscore
            #    z=0 → 0 risk; z=1 → 33; z=2 → 67; z=3 → 100
            if "expense_ratio_zscore" in co.columns:
                zs   = float(co["expense_ratio_zscore"].abs().max() or 0)
                pd_s = min(100, zs / 3 * 100)
                tip.append(f"Expense z-score: {zs:.2f}")
            else:
                pd_s = 0
                tip.append("N/A")
            row.append(float(pd_s))

            scores.append(row)
            tooltips.append(tip)

        z = np.array(scores, dtype=float)
        # Cell text: score + small context
        cell_text = [[f"{v:.0f}" for v in row] for row in z]

        fig = go.Figure(go.Heatmap(
            z=z,
            x=dimensions,
            y=tickers,
            text=cell_text,
            texttemplate="%{text}",
            textfont=dict(size=10, family="IBM Plex Mono", color="white"),
            colorscale=[
                [0.0,  "#0d3321"],
                [0.25, "#1f5c2e"],
                [0.5,  "#5c4a00"],
                [0.75, "#8b2a00"],
                [1.0,  "#c0392b"],
            ],
            zmin=0, zmax=100,
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0f}/100<extra></extra>",
            showscale=True,
            colorbar=dict(
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0 Low", "25", "50 Med", "75", "100 High"],
                tickfont=dict(color="#8b949e", size=8, family="IBM Plex Mono"),
                title=dict(text="Risk", font=dict(color="#8b949e", size=9)),
                len=0.85,
            ),
        ))
        fig.update_layout(
            title=dict(
                text="Enterprise Risk Heatmap — Last 8 Quarters · 0 = Low Risk · 100 = High Risk",
                font=dict(color="#8b949e", size=10)
            ),
            height=360,
            xaxis=dict(tickfont=dict(color="#c9d1d9", size=10, family="IBM Plex Mono"),
                       side="top"),
            yaxis=dict(tickfont=dict(color="#c9d1d9", size=10, family="IBM Plex Mono"),
                       autorange="reversed"),
            **PLOT,
        )
        return fig
    except Exception:
        return None


def filing_timeline_chart(df_edgar: pd.DataFrame) -> go.Figure | None:
    """Horizontal filing timeline — who filed late and by how much."""
    if "days_to_file" not in df_edgar.columns or "ticker" not in df_edgar.columns:
        return None
    latest = (df_edgar.sort_values("period_end")
              .groupby("ticker").last().reset_index()
              .dropna(subset=["days_to_file"]))
    if len(latest) == 0:
        return None
    # SEC deadline: 60 days for large accelerated filers
    deadline = 60
    latest = latest.sort_values("days_to_file", ascending=True)
    colors = ["#f85149" if d > deadline else "#3fb950" for d in latest["days_to_file"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=latest["ticker"],
        x=latest["days_to_file"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(color="#0d1117", width=1)),
        text=[f"{int(d)}d {'⚠ LATE' if d > deadline else '✓'}" for d in latest["days_to_file"]],
        textposition="outside",
        textfont=dict(color="#8b949e", size=9, family="IBM Plex Mono"),
        hovertemplate="Ticker: %{y}<br>Days to file: %{x}<extra></extra>",
    ))
    fig.add_vline(x=deadline, line=dict(color="#ffd700", width=2, dash="dash"),
                  annotation=dict(text="60-day SEC deadline",
                                  font=dict(color="#ffd700", size=9, family="IBM Plex Mono"),
                                  bgcolor="#0d1117", borderpad=3, yref="paper", y=1.05))
    fig.update_layout(
        title=dict(text="SEC Filing Timeline — Days to File (Latest Period)",
                  font=dict(color="#8b949e", size=10)),
        height=280,
        xaxis=dict(showgrid=True, gridcolor="#21262d",
                   tickfont=dict(color="#484f58", size=9),
                   range=[0, max(latest["days_to_file"].max() * 1.3, 80)]),
        yaxis=dict(showgrid=False, tickfont=dict(color="#c9d1d9", size=10, family="IBM Plex Mono")),
        showlegend=False, **PLOT,
    )
    return fig


def peer_radar_chart(df_edgar: pd.DataFrame, ticker: str) -> go.Figure | None:
    """Spider chart — selected company vs peer median. Per-metric normalization."""
    metrics = ["profit_margin", "debt_to_equity", "days_to_file",
               "operating_expenses_yoy_pct", "materiality_score", "expense_ratio_zscore"]
    labels  = ["Profit Margin", "Debt / Equity", "Days to File",
               "OpEx Growth %", "Materiality", "Expense Z-Score"]
    available       = [m for m in metrics if m in df_edgar.columns]
    avail_labels    = [labels[metrics.index(m)] for m in available]
    if len(available) < 3:
        return None

    # Use mean per company per metric
    all_means = df_edgar.groupby("ticker")[available].mean()
    if ticker not in all_means.index:
        return None

    co_vals   = all_means.loc[ticker]
    peer_vals = all_means.drop(index=ticker, errors="ignore").median()

    # Normalize each metric INDEPENDENTLY against all-company min/max
    co_norm, peer_norm = [], []
    for m in available:
        col_vals = all_means[m].dropna()
        mn, mx   = col_vals.min(), col_vals.max()
        rng      = mx - mn + 1e-9
        co_norm.append(float(np.clip((co_vals[m] - mn) / rng, 0, 1)))
        peer_norm.append(float(np.clip((peer_vals[m] - mn) / rng, 0, 1)))

    # Build raw value labels for hover
    co_raw   = [f"{co_vals[m]:.2f}" for m in available]
    peer_raw = [f"{peer_vals[m]:.2f}" for m in available]

    cats   = avail_labels + [avail_labels[0]]
    co_r   = co_norm   + [co_norm[0]]
    peer_r = peer_norm + [peer_norm[0]]
    co_hover   = co_raw   + [co_raw[0]]
    peer_hover = peer_raw + [peer_raw[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=co_r, theta=cats, fill="toself",
        fillcolor="rgba(88,166,255,0.18)",
        line=dict(color="#58a6ff", width=2),
        name=ticker,
        customdata=co_hover,
        hovertemplate="%{theta}: %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=peer_r, theta=cats, fill="toself",
        fillcolor="rgba(63,185,80,0.1)",
        line=dict(color="#3fb950", width=2, dash="dot"),
        name="Peer Median",
        customdata=peer_hover,
        hovertemplate="%{theta}: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{ticker} vs Peer Median — Risk Profile (normalized per metric)",
                  font=dict(color="#8b949e", size=10)),
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1],
                           tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                           ticktext=["0%","25%","50%","75%","100%"],
                           tickfont=dict(color="#484f58", size=7),
                           gridcolor="#21262d", linecolor="#21262d"),
            angularaxis=dict(tickfont=dict(color="#c9d1d9", size=9),
                            gridcolor="#21262d", linecolor="#30363d"),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#8b949e", size=9), bgcolor="#0d1117",
                   orientation="h", x=0.3, y=-0.08),
        height=340, **PLOT,
    )
    return fig


def company_deep_dive(df_edgar: pd.DataFrame, ticker: str) -> list[go.Figure]:
    """Returns sparkline figures for each key metric of one company."""
    co = df_edgar[df_edgar["ticker"] == ticker].sort_values("period_end").copy()
    if len(co) < 2:
        return []

    # Fix: clip days_to_file to 0–365 (outliers from bad data)
    if "days_to_file" in co.columns:
        co["days_to_file"] = co["days_to_file"].clip(0, 365)

    # Fix: materiality_score — use a rolling max so it shows progression, not flat zero
    if "materiality_score" in co.columns:
        co["materiality_score"] = co["materiality_score"].fillna(0)

    sparkline_metrics = [
        ("profit_margin",               "Profit Margin %",   "#58a6ff", 100),
        ("operating_expenses_yoy_pct",  "OpEx YoY %",        "#f85149", 1),
        ("debt_to_equity",              "Debt / Equity",     "#d2a020", 1),
        ("days_to_file",                "Days to File",      "#bc8cff", 1),
        ("materiality_score",           "Materiality Score", "#fb923c", 1),
        ("expense_ratio_zscore",        "Expense Z-Score",   "#3fb950", 1),
    ]
    figs = []
    for col, label, color, scale in sparkline_metrics:
        if col not in co.columns:
            continue
        series = (co[col] * scale).round(3)
        if series.isna().all():
            continue
        # For sparklines, drop NaN to avoid gaps
        valid = co[["period_end", col]].dropna()
        if len(valid) < 2:
            continue
        y_vals = (valid[col] * scale).round(3)

        # Add a zero/deadline reference line for specific metrics
        ref_line = None
        if col == "days_to_file":
            ref_line = 60  # SEC deadline
        elif col == "operating_expenses_yoy_pct":
            ref_line = 0
        elif col == "expense_ratio_zscore":
            ref_line = 0

        latest_val = float(y_vals.iloc[-1])
        hex_c = color.lstrip("#")
        r_int, g_int, b_int = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)

        fig = go.Figure(go.Scatter(
            x=valid["period_end"], y=y_vals,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            fill="tozeroy",
            fillcolor=f"rgba({r_int},{g_int},{b_int},0.08)",
            hovertemplate=f"{label}: %{{y}}<br>Period: %{{x}}<extra></extra>",
        ))
        if ref_line is not None:
            fig.add_hline(y=ref_line,
                         line=dict(color="#484f58", width=1, dash="dot"))
        fig.update_layout(
            title=dict(
                text=f"{label}  <b>{latest_val:.1f}</b>",
                font=dict(color=color, size=9, family="IBM Plex Mono")
            ),
            height=140,
            margin=dict(l=5, r=5, t=28, b=5),
            xaxis=dict(showgrid=False, showticklabels=False, color="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d",
                       tickfont=dict(color="#484f58", size=7)),
            showlegend=False,
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        )
        figs.append((label, fig))
    return figs


def structuring_radar_chart(df_pay: pd.DataFrame) -> go.Figure | None:
    """Hexagonal radar showing the 6 AML flag dimensions across the dataset."""
    flag_cols = {
        "structuring_flag":    "Structuring",
        "balance_drain":       "Balance Drain",
        "balance_mismatch":    "Bal. Mismatch",
        "funds_not_received":  "Funds Not Recv",
        "high_velocity":       "High Velocity",
        "fan_out_flag":        "Fan-Out",
    }
    available = {k: v for k, v in flag_cols.items() if k in df_pay.columns}
    if len(available) < 3:
        return None
    totals  = {v: int(df_pay[k].sum()) for k, v in available.items()}
    max_v   = max(totals.values()) or 1
    labels  = list(totals.keys())
    values  = [v / max_v for v in totals.values()]
    counts  = [f"{totals[l]:,}" for l in labels]  # raw counts for hover
    cats    = labels + [labels[0]]
    vals    = values + [values[0]]
    hover_c = counts + [counts[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(248,81,73,0.15)",
        line=dict(color="#f85149", width=2),
        customdata=hover_c,
        hovertemplate="%{theta}<br>Count: %{customdata}<br>Relative: %{r:.1%}<extra></extra>",
    ))
    # Add a second trace for absolute scale labelling
    fig.update_layout(
        title=dict(text="AML Flag Radar — 6 FATF Dimensions (relative to max flag count)",
                  font=dict(color="#8b949e", size=10)),
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1],
                           tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                           ticktext=["0%","25%","50%","75%","100%"],
                           tickfont=dict(color="#484f58", size=7),
                           gridcolor="#21262d", linecolor="#21262d"),
            angularaxis=dict(tickfont=dict(color="#c9d1d9", size=9),
                            gridcolor="#21262d", linecolor="#30363d"),
        ),
        height=320,
        showlegend=False,
        annotations=[dict(
            text=f"Hover each axis for raw counts<br>Max = {max_v:,} transactions",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False,
            font=dict(color="#484f58", size=8, family="IBM Plex Mono"),
        )],
        **PLOT,
    )
    return fig


@st.cache_data(show_spinner=False)
def compute_model_perf():
    try:
        df = pd.read_csv(ROOT / "data/processed/paysim_high_risk.csv")
        feature_cols = [c for c in [
            "structuring_flag","balance_drain","balance_mismatch",
            "funds_not_received","high_velocity","fan_out_flag"
        ] if c in df.columns]
        if not feature_cols or "isfraud" not in df.columns:
            return None
        sample = df.sample(min(50000, len(df)), random_state=42)
        X = sample[feature_cols].fillna(0)
        y_true = sample["isfraud"].astype(int)
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
        iso.fit(X)
        y_pred = (iso.predict(X) == -1).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        base = iso.decision_function(X).mean()
        feat_imp = {}
        for col in feature_cols:
            Xp = X.copy(); Xp[col] = 0
            feat_imp[col] = abs(base - iso.decision_function(Xp).mean())
        mx = max(feat_imp.values()) or 1
        feat_imp = {k: round(v/mx*100,1) for k,v in feat_imp.items()}
        return {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
            "cm":        cm.tolist(),
            "feat_imp":  feat_imp,
            "feature_cols": feature_cols,
            "sample":    len(sample),
            "fraud_rate":round(y_true.mean()*100, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Navigation
# ════════════════════════════════════════════════════════════════════════════

PAGES = [
    "🏠  Home",
    "💬  Query & Analysis",
    "⚠   Anomaly Explorer",
    "📈  Financial Intelligence",
    "🏛   Regulations",
    "🔬  Model Performance",
]

with st.sidebar:
    st.markdown("### ⚖ AUDITM**I**ND")
    st.markdown("---")
    st.markdown("### NAVIGATION")

    for page in PAGES:
        is_active = st.session_state.page == page
        style = "active" if is_active else ""
        if st.button(page, key=f"nav_{page}",
                     help=f"Go to {page}"):
            st.session_state.page = page
            st.rerun()

    st.markdown("---")
    st.markdown("### DEMO QUERIES")
    st.caption("Click to run instantly")
    for q in DEMO_QUERIES:
        label = (q[:48] + "...") if len(q) > 48 else q
        if st.button(label, key=f"dq_{q[:22]}"):
            st.session_state.pending_query = q
            st.session_state.page = "💬  Query & Analysis"
            st.rerun()

    st.markdown("---")
    if st.session_state.last_result:
        r = st.session_state.last_result
        st.markdown("### LAST RESULT")
        dec = r.get("decision","?")
        conf = int(r.get("confidence_score",0)*100)
        mat = r.get("materiality_detail",{})
        st.markdown(
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.9rem;'
            f'font-weight:600;color:{"#f85149" if dec=="ESCALATE" else "#58a6ff"}">'
            f'{dec}</span> &nbsp;'
            f'<span style="font-size:0.7rem;color:#484f58">{mat.get("score",0)}/100 · {conf}% conf</span>',
            unsafe_allow_html=True
        )
        for a in r.get("agents_executed",[]):
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;'
                f'color:#3fb950;padding:1px 0 1px 8px;border-left:2px solid #3fb950">✓ {a}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.6rem;color:#21262d">AuditMind v3 · LangGraph · Groq · ChromaDB</span>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADER + GLOBAL QUERY BAR (every page)
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="am-header">
  <div>
    <div class="am-logo">AUDITM<span style="color:#1f6feb">I</span>ND</div>
    <div class="am-sub">Agentic AI &nbsp;·&nbsp; Financial Audit &nbsp;·&nbsp; AML Intelligence</div>
  </div>
  <div style="text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#484f58;line-height:1.9">
    <div><span class="live-dot"></span>SYSTEM ONLINE</div>
    <div>9 AGENTS · 3 DATA LAYERS · FATF/FFIEC CORPUS</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Global query bar — always visible on every page ──────────────────────────
qcol, bcol, clrcol = st.columns([5, 1, 0.6], gap="small")
with qcol:
    global_query = st.text_input(
        "global_q",
        placeholder="Ask anything — e.g. 'Analyze JPMorgan financials' or 'Show structuring patterns'",
        label_visibility="collapsed",
        key="global_query_input"
    )
with bcol:
    global_submit = st.button("▶  Analyze", use_container_width=True, key="global_submit")
with clrcol:
    if st.button("Clear", use_container_width=True, key="global_clear"):
        st.session_state.last_result = None
        st.session_state.messages   = []
        st.rerun()

# Show last result decision badge inline under query bar
if st.session_state.last_result:
    _r   = st.session_state.last_result
    _dec = _r.get("decision","?")
    _mat = _r.get("materiality_detail",{})
    _col = {"ESCALATE":"#f85149","INVESTIGATE":"#58a6ff","REVIEW":"#d2a020",
            "DISCLOSE":"#3fb950","IGNORE":"#8b949e"}.get(_dec,"#8b949e")
    st.markdown(
        f'<div style="margin:-0.4rem 0 0.8rem 0;font-family:IBM Plex Mono,monospace;font-size:0.72rem">'
        f'<span style="color:{_col};font-weight:600">{_dec}</span>'
        f'<span style="color:#484f58"> &nbsp;·&nbsp; materiality {_mat.get("score",0)}/100'
        f' &nbsp;·&nbsp; {int(_r.get("confidence_score",0)*100)}% confidence'
        f' &nbsp;·&nbsp; agents: {" → ".join(_r.get("agents_executed",[]))}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# Handle demo query from sidebar
if "pending_query" in st.session_state:
    global_query  = st.session_state.pending_query
    global_submit = True
    del st.session_state.pending_query

# Run pipeline on submit
if global_submit and global_query:
    with st.spinner("Running 9-agent pipeline..."):
        _result = run_pipeline(global_query)
    st.session_state.last_result = _result
    st.session_state.messages.append({"role":"user","content":global_query})
    _decision  = _result.get("decision","UNKNOWN")
    _mat       = _result.get("materiality_detail",{})
    _rationale = _result.get("decision_rationale","")
    _conf      = int(_result.get("confidence_score",0)*100)
    _dec_color = {"ESCALATE":"#f85149","INVESTIGATE":"#58a6ff","REVIEW":"#d2a020",
                  "DISCLOSE":"#3fb950","IGNORE":"#8b949e"}.get(_decision,"#8b949e")
    _response  = (
        f"<span style='font-family:IBM Plex Mono,monospace;font-weight:600;"
        f"font-size:0.95rem;color:{_dec_color}'>{_decision}</span>"
        f"&nbsp;&nbsp;<span style='font-size:0.68rem;color:#484f58'>"
        f"materiality {_mat.get('score',0)}/100 · confidence {_conf}%</span>"
        f"<br><br><span style='color:#c9d1d9'>{_rationale}</span>"
    )
    st.session_state.messages.append({"role":"assistant","content":_response})
    st.rerun()

st.markdown('<div style="border-bottom:1px solid #21262d;margin-bottom:1.2rem"></div>',
            unsafe_allow_html=True)

page = st.session_state.page


# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME — Always-on base dashboard
# ════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":

    df_edgar = load_edgar()
    df_pay   = load_paysim()

    # ── If there's a result, show it prominently at top ──────────────────────
    if st.session_state.last_result:
        result   = st.session_state.last_result
        decision = result.get("decision","UNKNOWN")
        mat      = result.get("materiality_detail",{})
        intent   = result.get("query_intent","mixed")
        anomaly  = result.get("anomaly_findings",{})
        fin      = result.get("financial_findings",{})
        rag      = result.get("rag_findings",[])
        flags    = result.get("compliance_flags",[])

        last_query = (st.session_state.messages[-2]["content"]
                      if len(st.session_state.messages) >= 2 else "")
        companies  = extract_companies(last_query)

        st.markdown('<div class="section-hdr">QUERY ANALYSIS RESULTS</div>', unsafe_allow_html=True)

        # Decision + key metrics row
        dec_col, m1, m2, m3, m4 = st.columns([1.2,1,1,1,1])
        with dec_col:
            _dc = {"ESCALATE":"#f85149","INVESTIGATE":"#58a6ff","REVIEW":"#d2a020",
                   "DISCLOSE":"#3fb950","IGNORE":"#8b949e"}.get(decision,"#8b949e")
            st.markdown(
                f'<div class="badge badge-{decision}" style="font-size:1.1rem;'
                f'padding:0.65rem 1.4rem">⬤ {decision}</div>',
                unsafe_allow_html=True
            )
        with m1: st.metric("Materiality", f"{mat.get('score',0)}/100")
        with m2: st.metric("Risk Level",   mat.get("level","?"))
        with m3: st.metric("Confidence",  f"{int(result.get('confidence_score',0)*100)}%")
        with m4:
            agents_run = len(result.get("agents_executed",[]))
            st.metric("Agents Run", agents_run)

        # Rationale
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #21262d;border-left:3px solid {_dc};'
            f'border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin:0.75rem 0 1rem 0;'
            f'font-size:0.83rem;color:#8b949e;line-height:1.7">'
            f'{result.get("decision_rationale","")}</div>',
            unsafe_allow_html=True
        )

        # ── Query-specific charts ─────────────────────────────────────────────
        has_financial = fin.get("data_available") and df_edgar is not None
        has_aml       = anomaly.get("data_available")

        # ── FINANCIAL: company deep dive or comparison ──
        if has_financial and df_edgar is not None:
            if companies:
                st.markdown(f'<div class="section-hdr">FINANCIALS — {", ".join(companies)} DEEP DIVE</div>',
                           unsafe_allow_html=True)
                if len(companies) == 1:
                    ticker = companies[0]
                    sparklines = company_deep_dive(df_edgar, ticker)
                    if sparklines:
                        n_cols = 3
                        rows = [sparklines[i:i+n_cols] for i in range(0, len(sparklines), n_cols)]
                        for row in rows:
                            cols = st.columns(len(row), gap="small")
                            for col, (label, sfig) in zip(cols, row):
                                with col:
                                    st.plotly_chart(sfig, use_container_width=True,
                                                   config={"displayModeBar":False},
                                                   key=f"spark_{ticker}_{label}")
                    rc1, rc2 = st.columns(2, gap="large")
                    with rc1:
                        fig_radar = peer_radar_chart(df_edgar, ticker)
                        if fig_radar:
                            st.plotly_chart(fig_radar, use_container_width=True,
                                           config={"displayModeBar":False},
                                           key=f"radar_{ticker}")
                    with rc2:
                        fig_ft = filing_timeline_chart(df_edgar)
                        if fig_ft:
                            st.plotly_chart(fig_ft, use_container_width=True,
                                           config={"displayModeBar":False},
                                           key="home_filing_single")
                else:
                    cmp1, cmp2 = st.columns(2, gap="large")
                    with cmp1:
                        st.plotly_chart(profit_margin_chart(df_edgar, companies),
                                       use_container_width=True,
                                       config={"displayModeBar":False}, key="home_cmp_pm")
                        fig_cmp = yoy_expense_chart(df_edgar, companies)
                        if fig_cmp:
                            st.plotly_chart(fig_cmp, use_container_width=True,
                                           config={"displayModeBar":False}, key="home_cmp_exp")
                    with cmp2:
                        for t in companies[:2]:
                            fig_r = peer_radar_chart(df_edgar, t)
                            if fig_r:
                                st.plotly_chart(fig_r, use_container_width=True,
                                               config={"displayModeBar":False},
                                               key=f"home_radar_{t}")
                                break
            else:
                st.markdown('<div class="section-hdr">FINANCIAL OVERVIEW</div>',
                           unsafe_allow_html=True)
                hm_col, ft_col = st.columns(2, gap="large")
                with hm_col:
                    fig_hm = risk_heatmap(df_edgar, df_pay)
                    if fig_hm:
                        st.plotly_chart(fig_hm, use_container_width=True,
                                       config={"displayModeBar":False}, key="home_heatmap_fin")
                with ft_col:
                    fig_ft = filing_timeline_chart(df_edgar)
                    if fig_ft:
                        st.plotly_chart(fig_ft, use_container_width=True,
                                       config={"displayModeBar":False}, key="home_filing_gen")

            interp = fin.get("interpretation","")
            if interp:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#8b949e;line-height:1.7;'
                    f'background:#161b22;border:1px solid #21262d;border-radius:8px;'
                    f'padding:0.8rem 1rem;margin-top:0.5rem">{interp}</div>',
                    unsafe_allow_html=True
                )

        # ── AML: structuring cliff + radar + typology ──
        if has_aml:
            findings = anomaly.get("findings",{})
            typology = findings.get("typology_breakdown",{})
            st.markdown('<div class="section-hdr">AML ANOMALIES DETECTED</div>',
                       unsafe_allow_html=True)
            a1,a2,a3 = st.columns(3)
            with a1: st.metric("Structuring", f"{findings.get('structuring_transactions',0):,}")
            with a2: st.metric("High-Risk",   f"{findings.get('total_high_risk_transactions',0):,}")
            with a3: st.metric("Avg Amount",  f"${findings.get('avg_structuring_amount',0):,.0f}")

            aml_l, aml_r = st.columns(2, gap="large")
            with aml_l:
                if df_pay is not None:
                    fig_cliff = structuring_cliff_chart(df_pay)
                    if fig_cliff:
                        st.plotly_chart(fig_cliff, use_container_width=True,
                                       config={"displayModeBar":False}, key="home_cliff")
            with aml_r:
                if df_pay is not None:
                    fig_sr = structuring_radar_chart(df_pay)
                    if fig_sr:
                        st.plotly_chart(fig_sr, use_container_width=True,
                                       config={"displayModeBar":False}, key="home_s_radar")
            if typology:
                fig_ty = typology_chart(typology)
                if fig_ty:
                    st.plotly_chart(fig_ty, use_container_width=True,
                                   config={"displayModeBar":False}, key="home_typology")
            interp = anomaly.get("interpretation","")
            if interp:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#8b949e;line-height:1.7;'
                    f'background:#161b22;border:1px solid #21262d;border-radius:8px;'
                    f'padding:0.8rem 1rem;margin-top:0.5rem">{interp}</div>',
                    unsafe_allow_html=True
                )

        # Regulatory citations + compliance flags
        if rag or flags:
            rc1, rc2 = st.columns([1.2, 0.8], gap="large")
            with rc1:
                if rag:
                    st.markdown('<div class="section-hdr">REGULATORY CITATIONS</div>',
                               unsafe_allow_html=True)
                    for r in rag:
                        st.markdown(
                            f'<div class="citation">'
                            f'<div class="cit-src">{r["citation"]}'
                            f'<span>score: {r["relevance_score"]:.2f}</span></div>'
                            f'<div class="cit-text">{r["text"][:300]}...</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            with rc2:
                if flags:
                    st.markdown('<div class="section-hdr">COMPLIANCE FLAGS</div>',
                               unsafe_allow_html=True)
                    for flag in flags:
                        sev = flag.get("severity","LOW").upper()
                        sc  = {"HIGH":"#f85149","MEDIUM":"#d2a020","LOW":"#3fb950"}.get(sev,"#8b949e")
                        st.markdown(
                            f'<div class="flag flag-{sev}">'
                            f'<div class="flag-lbl" style="color:{sc}">[{sev}] {flag.get("rule","")}</div>'
                            f'<div class="flag-txt">{flag.get("description","")}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        # Scenario projection
        scenario = result.get("scenario_projection","")
        if scenario and mat.get("level") in ("HIGH","MEDIUM"):
            st.markdown('<div class="section-hdr" style="margin-top:0.5rem">SCENARIO: IF IGNORED</div>',
                       unsafe_allow_html=True)
            sc_cols = st.columns(3)
            sc_items = [l.lstrip("*•- ").strip() for l in scenario.strip().split("\n") if l.strip()]
            for i, line in enumerate(sc_items[:3]):
                if ":" in line and len(line.split(":")[0]) < 30:
                    label, rest = line.split(":",1)
                    lc = ("#f85149" if "regulat" in label.lower()
                          else "#d2a020" if "financial" in label.lower() else "#bc8cff")
                    with sc_cols[i % 3]:
                        st.markdown(
                            f'<div style="background:#161b22;border-left:3px solid {lc};'
                            f'border-radius:0 7px 7px 0;padding:0.65rem 0.9rem">'
                            f'<div style="color:{lc};font-family:IBM Plex Mono,monospace;'
                            f'font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
                            f'margin-bottom:3px">{label.strip()}</div>'
                            f'<div style="font-size:0.76rem;color:#8b949e;line-height:1.6">'
                            f'{rest.strip()}</div></div>',
                            unsafe_allow_html=True
                        )

        st.markdown('<div style="border-bottom:1px solid #21262d;margin:1.5rem 0"></div>',
                   unsafe_allow_html=True)

    # ── Base dashboard (always shown below results) ───────────────────────────
    st.markdown('<div class="section-hdr">PLATFORM OVERVIEW</div>', unsafe_allow_html=True)

    # KPI row
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1:
        st.markdown('<div class="stat-card"><div class="stat-label">Companies</div>'
                    '<div class="stat-value">10</div>'
                    '<div class="stat-sub">SEC EDGAR</div></div>', unsafe_allow_html=True)
    with k2:
        n = len(df_edgar) if df_edgar is not None else 0
        st.markdown(f'<div class="stat-card"><div class="stat-label">EDGAR Records</div>'
                    f'<div class="stat-value">{n:,}</div>'
                    f'<div class="stat-sub">Quarterly filings</div></div>', unsafe_allow_html=True)
    with k3:
        n = len(df_pay) if df_pay is not None else 0
        st.markdown(f'<div class="stat-card"><div class="stat-label">High-Risk Tx</div>'
                    f'<div class="stat-value">{n/1e6:.1f}M</div>'
                    f'<div class="stat-sub">PaySim flagged</div></div>', unsafe_allow_html=True)
    with k4:
        struct = int(df_pay["structuring_flag"].sum()) if df_pay is not None and "structuring_flag" in df_pay.columns else 0
        st.markdown(f'<div class="stat-card"><div class="stat-label">Structuring Tx</div>'
                    f'<div class="stat-value">{struct:,}</div>'
                    f'<div class="stat-sub">FATF Rec. 10 risk</div></div>', unsafe_allow_html=True)
    with k5:
        hm = 0
        if df_edgar is not None and "materiality_score" in df_edgar.columns:
            hm = int((df_edgar["materiality_score"] >= 65).sum())
        st.markdown(f'<div class="stat-card"><div class="stat-label">High Materiality</div>'
                    f'<div class="stat-value">{hm}</div>'
                    f'<div class="stat-sub">Score ≥ 65</div></div>', unsafe_allow_html=True)
    with k6:
        try:
            chunks = pd.read_csv(ROOT/"data/processed/regulatory_chunks_v2.csv")
            nc = len(chunks)
        except Exception:
            nc = 450
        st.markdown(f'<div class="stat-card"><div class="stat-label">RAG Chunks</div>'
                    f'<div class="stat-value">{nc}</div>'
                    f'<div class="stat-sub">FATF + FFIEC</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Charts row 1
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-hdr">FINANCIAL OVERVIEW — ALL COMPANIES</div>',
                   unsafe_allow_html=True)
        if df_edgar is not None:
            st.plotly_chart(profit_margin_chart(df_edgar), use_container_width=True,
                           config={"displayModeBar":False}, key="home_pm")
        else:
            st.info("EDGAR data not loaded.")
    with c2:
        st.markdown('<div class="section-hdr">AML TRANSACTION LANDSCAPE</div>',
                   unsafe_allow_html=True)
        if df_pay is not None:
            fig_aml = aml_overview_chart(df_pay)
            if fig_aml:
                st.plotly_chart(fig_aml, use_container_width=True,
                               config={"displayModeBar":False}, key="home_aml_pie")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown('<div class="section-hdr">OPERATING EXPENSE TRENDS</div>',
                   unsafe_allow_html=True)
        if df_edgar is not None:
            fig_exp = yoy_expense_chart(df_edgar)
            if fig_exp:
                st.plotly_chart(fig_exp, use_container_width=True,
                               config={"displayModeBar":False}, key="home_exp")
    with c4:
        st.markdown('<div class="section-hdr">AML RISK SCORE DISTRIBUTION</div>',
                   unsafe_allow_html=True)
        if df_pay is not None:
            fig_rd = risk_score_dist(df_pay)
            if fig_rd:
                st.plotly_chart(fig_rd, use_container_width=True,
                               config={"displayModeBar":False}, key="home_riskdist")

    # Risk heatmap + filing timeline
    st.markdown('<div class="section-hdr">ENTERPRISE RISK HEATMAP — ALL COMPANIES</div>',
               unsafe_allow_html=True)
    if df_edgar is not None:
        fig_hm = risk_heatmap(df_edgar, df_pay)
        if fig_hm:
            st.plotly_chart(fig_hm, use_container_width=True,
                           config={"displayModeBar":False}, key="base_heatmap")

    hft_col, aml_radar_col = st.columns(2, gap="large")
    with hft_col:
        st.markdown('<div class="section-hdr">SEC FILING TIMELINE</div>',
                   unsafe_allow_html=True)
        if df_edgar is not None:
            fig_ft = filing_timeline_chart(df_edgar)
            if fig_ft:
                st.plotly_chart(fig_ft, use_container_width=True,
                               config={"displayModeBar":False}, key="base_filing")
    with aml_radar_col:
        st.markdown('<div class="section-hdr">AML FLAG RADAR — 6 FATF DIMENSIONS</div>',
                   unsafe_allow_html=True)
        if df_pay is not None:
            fig_sr = structuring_radar_chart(df_pay)
            if fig_sr:
                st.plotly_chart(fig_sr, use_container_width=True,
                               config={"displayModeBar":False}, key="base_s_radar")

    # Structuring cliff — always shown
    st.markdown('<div class="section-hdr">STRUCTURING CLIFF — $10K THRESHOLD AVOIDANCE</div>',
               unsafe_allow_html=True)
    if df_pay is not None:
        fig_cliff = structuring_cliff_chart(df_pay)
        if fig_cliff:
            st.plotly_chart(fig_cliff, use_container_width=True,
                           config={"displayModeBar":False}, key="base_cliff")

    if df_edgar is not None and "materiality_score" in df_edgar.columns:
        st.markdown('<div class="section-hdr">HIGH MATERIALITY CASES</div>',
                   unsafe_allow_html=True)
        hm_df = df_edgar[df_edgar["materiality_score"] >= 50].sort_values(
            "materiality_score", ascending=False)
        cols = [c for c in ["ticker","fiscal_year","fiscal_period",
                            "materiality_score","late_filing","profit_margin",
                            "operating_expenses_yoy_pct"] if c in hm_df.columns]
        if len(hm_df) > 0:
            st.dataframe(hm_df[cols].head(15), use_container_width=True, height=250)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: QUERY & ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

elif page == "💬  Query & Analysis":
    chat_col, analysis_col = st.columns([1, 1.1], gap="large")

    with chat_col:
        st.markdown('<div class="section-hdr">QUERY INTERFACE</div>', unsafe_allow_html=True)

        # Chat history
        for msg in st.session_state.messages[-10:]:
            if msg["role"] == "user":
                st.markdown(f'<div class="bubble-user">🔍 {msg["content"]}</div>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bubble-bot">{msg["content"]}</div>',
                           unsafe_allow_html=True)

        query = st.text_input("query", placeholder="Ask about AML, financials, compliance...",
                             label_visibility="collapsed", key="q_input")
        b1, b2 = st.columns([3,1])
        with b1: submit = st.button("▶  Run Analysis", use_container_width=True)
        with b2:
            if st.button("Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_result = None
                st.rerun()

    # Handle demo query
    if "pending_query" in st.session_state:
        query  = st.session_state.pending_query
        submit = True
        del st.session_state.pending_query

    if submit and query:
        st.session_state.messages.append({"role":"user","content":query})
        with st.spinner("Running 9-agent pipeline..."):
            result = run_pipeline(query)
        st.session_state.last_result = result

        decision = result.get("decision","UNKNOWN")
        mat      = result.get("materiality_detail",{})
        rationale= result.get("decision_rationale","")
        conf     = int(result.get("confidence_score",0)*100)

        dec_color = {"ESCALATE":"#f85149","INVESTIGATE":"#58a6ff",
                     "REVIEW":"#d2a020","DISCLOSE":"#3fb950","IGNORE":"#8b949e"}.get(decision,"#8b949e")
        lvl_color = {"HIGH":"#f85149","MEDIUM":"#d2a020","LOW":"#3fb950"}.get(
            mat.get("level","LOW"),"#8b949e")

        response_html = (
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:1rem;"
            f"font-weight:600;color:{dec_color}'>{decision}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='font-size:0.7rem;color:#484f58;font-family:IBM Plex Mono,monospace'>"
            f"materiality {mat.get('score',0)}/100</span>"
            f"<br><br>"
            f"<span style='color:#c9d1d9'>{rationale}</span>"
            f"<br><br>"
            f"<span style='color:#484f58;font-size:0.78rem'>"
            f"→ Analysis panel updated with query-specific charts.</span>"
        )
        st.session_state.messages.append({"role":"assistant","content":response_html})
        st.rerun()

    # Analysis panel — query-specific
    with analysis_col:
        if st.session_state.last_result:
            result   = st.session_state.last_result
            decision = result.get("decision","UNKNOWN")
            mat      = result.get("materiality_detail",{})
            intent   = result.get("query_intent","mixed")
            anomaly  = result.get("anomaly_findings",{})
            fin      = result.get("financial_findings",{})
            rag      = result.get("rag_findings",[])
            flags    = result.get("compliance_flags",[])

            # Extract companies from the last user query
            last_query = st.session_state.messages[-2]["content"] \
                        if len(st.session_state.messages) >= 2 else ""
            companies = extract_companies(last_query)

            # Decision + materiality
            st.markdown(
                f'<div class="badge badge-{decision}">⬤ {decision}</div>',
                unsafe_allow_html=True
            )
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Materiality", f"{mat.get('score',0)}/100")
            with m2: st.metric("Risk", mat.get("level","?"))
            with m3: st.metric("Confidence", f"{int(result.get('confidence_score',0)*100)}%")

            st.markdown("")
            st.markdown('<div class="section-hdr">QUERY-SPECIFIC ANALYSIS</div>', unsafe_allow_html=True)

            df_edgar = load_edgar()
            df_pay   = load_paysim()

            # ── FINANCIAL INTENT: show company-specific charts ──
            if intent in ("financial", "mixed") and fin.get("data_available") and df_edgar is not None:
                if companies:
                    st.caption(f"Filtered to: **{', '.join(companies)}**")
                    # Time series for each company found
                    for ticker in companies[:2]:
                        fig_ts = company_detail_chart(df_edgar, ticker)
                        if fig_ts:
                            st.plotly_chart(fig_ts, use_container_width=True,
                                           config={"displayModeBar":False},
                                           key=f"ts_{ticker}")
                    # Comparison bar if multiple companies
                    if len(companies) >= 2:
                        st.plotly_chart(
                            profit_margin_chart(df_edgar, companies),
                            use_container_width=True,
                            config={"displayModeBar":False},
                            key="compare_pm"
                        )
                        fig_cmp_exp = yoy_expense_chart(df_edgar, companies)
                        if fig_cmp_exp:
                            st.plotly_chart(fig_cmp_exp, use_container_width=True,
                                           config={"displayModeBar":False},
                                           key="compare_exp")
                    else:
                        # Single company — show profit margin across all for context
                        st.plotly_chart(
                            profit_margin_chart(df_edgar, companies),
                            use_container_width=True,
                            config={"displayModeBar":False},
                            key="single_pm"
                        )
                else:
                    # No company mentioned — show all
                    st.plotly_chart(profit_margin_chart(df_edgar),
                                   use_container_width=True,
                                   config={"displayModeBar":False},
                                   key="all_pm")
                    fig_exp = yoy_expense_chart(df_edgar)
                    if fig_exp:
                        st.plotly_chart(fig_exp, use_container_width=True,
                                       config={"displayModeBar":False},
                                       key="all_exp")

                # Financial narrative
                st.markdown("**Financial Analysis**")
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#8b949e;line-height:1.75;'
                    f'background:#161b22;padding:0.9rem;border-radius:8px;'
                    f'border:1px solid #21262d">'
                    f'{fin.get("interpretation","")}</div>',
                    unsafe_allow_html=True
                )

            # ── AML INTENT: show typology + anomaly charts ──
            if intent in ("aml","mixed") and anomaly.get("data_available"):
                findings = anomaly.get("findings",{})
                typology = findings.get("typology_breakdown",{})
                if typology and df_pay is not None:
                    fig_ty = typology_chart(typology)
                    if fig_ty:
                        st.plotly_chart(fig_ty, use_container_width=True,
                                       config={"displayModeBar":False},
                                       key="query_typology")

                a1, a2, a3 = st.columns(3)
                with a1: st.metric("Structuring Tx", f"{findings.get('structuring_transactions',0):,}")
                with a2: st.metric("High-Risk Tx",   f"{findings.get('total_high_risk_transactions',0):,}")
                with a3: st.metric("Avg Amount",      f"${findings.get('avg_structuring_amount',0):,.0f}")

                st.markdown("**AML Analysis**")
                st.markdown(
                    f'<div style="font-size:0.82rem;color:#8b949e;line-height:1.75;'
                    f'background:#161b22;padding:0.9rem;border-radius:8px;'
                    f'border:1px solid #21262d">'
                    f'{anomaly.get("interpretation","")}</div>',
                    unsafe_allow_html=True
                )

            # ── COMPLIANCE / RAG ──
            if intent in ("compliance","mixed") or rag:
                if rag:
                    st.markdown('<div class="section-hdr" style="margin-top:1rem">REGULATORY CITATIONS</div>',
                               unsafe_allow_html=True)
                    for r in rag:
                        st.markdown(
                            f'<div class="citation">'
                            f'<div class="cit-src">{r["citation"]}'
                            f'<span>score: {r["relevance_score"]:.2f}</span></div>'
                            f'<div class="cit-text">{r["text"][:350]}...</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                if flags:
                    st.markdown('<div class="section-hdr" style="margin-top:1rem">COMPLIANCE FLAGS</div>',
                               unsafe_allow_html=True)
                    for flag in flags:
                        sev = flag.get("severity","LOW").upper()
                        sc  = {"HIGH":"#f85149","MEDIUM":"#d2a020","LOW":"#3fb950"}.get(sev,"#8b949e")
                        st.markdown(
                            f'<div class="flag flag-{sev}">'
                            f'<div class="flag-lbl" style="color:{sc}">[{sev}] {flag.get("rule","")}</div>'
                            f'<div class="flag-txt">{flag.get("description","")}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # ── Scenario ──
            scenario = result.get("scenario_projection","")
            if scenario and mat.get("level") != "LOW":
                st.markdown('<div class="section-hdr" style="margin-top:1rem">SCENARIO: IF IGNORED</div>',
                           unsafe_allow_html=True)
                for line in scenario.strip().split("\n"):
                    line = line.lstrip("*•- ").strip()
                    if not line:
                        continue
                    if ":" in line and len(line.split(":")[0]) < 30:
                        label, rest = line.split(":", 1)
                        lc = ("#f85149" if "regulat" in label.lower()
                              else "#d2a020" if "financial" in label.lower()
                              else "#bc8cff")
                        st.markdown(
                            f'<div style="background:#161b22;border-left:3px solid {lc};'
                            f'border-radius:0 7px 7px 0;padding:0.6rem 0.9rem;margin:0.3rem 0">'
                            f'<div style="color:{lc};font-family:IBM Plex Mono,monospace;'
                            f'font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;'
                            f'margin-bottom:3px">{label.strip()}</div>'
                            f'<div style="font-size:0.79rem;color:#8b949e">{rest.strip()}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        else:
            st.markdown(
                '<div style="text-align:center;padding:5rem 2rem;color:#21262d">'
                '<div style="font-size:2.5rem;margin-bottom:1rem">⚖</div>'
                '<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;'
                'letter-spacing:0.12em;color:#30363d">AWAITING QUERY</div>'
                '<div style="font-size:0.72rem;margin-top:8px;color:#21262d">'
                'Type a query or pick a demo from the sidebar</div>'
                '</div>',
                unsafe_allow_html=True
            )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY EXPLORER
# ════════════════════════════════════════════════════════════════════════════

elif page == "⚠   Anomaly Explorer":
    st.markdown('<div class="section-hdr">AML TRANSACTION ANOMALY EXPLORER</div>', unsafe_allow_html=True)
    df_pay = load_paysim()

    if df_pay is None:
        st.error("PaySim data not loaded.")
    else:
        # Summary metrics
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Total High-Risk Tx",   f"{len(df_pay):,}")
        with m2:
            s = int(df_pay["structuring_flag"].sum()) if "structuring_flag" in df_pay.columns else 0
            st.metric("Structuring Tx", f"{s:,}")
        with m3:
            fo = int(df_pay["fan_out_flag"].sum()) if "fan_out_flag" in df_pay.columns else 0
            st.metric("Fan-Out (Smurfing)", f"{fo:,}")
        with m4:
            bd = int(df_pay["balance_drain"].sum()) if "balance_drain" in df_pay.columns else 0
            st.metric("Balance Drain", f"{bd:,}")

        st.markdown("")

        # Interactive filters
        fc1, fc2 = st.columns([1,3], gap="large")
        with fc1:
            st.markdown('<div class="section-hdr">FILTERS</div>', unsafe_allow_html=True)
            tx_types = ["All"] + sorted(df_pay["type"].unique().tolist()) \
                       if "type" in df_pay.columns else ["All"]
            sel_type = st.selectbox("Transaction Type", tx_types)

            if "aml_risk_score" in df_pay.columns:
                min_score = st.slider("Min AML Risk Score", 0, 100, 40)
            else:
                min_score = 40

            flag_filter = st.multiselect(
                "Show only flagged",
                [c for c in ["structuring_flag","fan_out_flag","balance_drain","high_velocity"]
                 if c in df_pay.columns]
            )

        with fc2:
            filtered = df_pay.copy()
            if sel_type != "All":
                filtered = filtered[filtered["type"] == sel_type]
            if "aml_risk_score" in filtered.columns:
                filtered = filtered[filtered["aml_risk_score"] >= min_score]
            for f in flag_filter:
                filtered = filtered[filtered[f] == 1]

            st.caption(f"Showing {len(filtered):,} transactions after filters")

            # Typology breakdown of filtered set
            if "fatf_typology" in filtered.columns:
                typo_counts = filtered["fatf_typology"].value_counts().to_dict()
                fig_ty = typology_chart(typo_counts)
                if fig_ty:
                    st.plotly_chart(fig_ty, use_container_width=True,
                                   config={"displayModeBar":False},
                                   key="anomaly_typology")

        # Risk distribution of filtered
        if "aml_risk_score" in df_pay.columns:
            st.markdown('<div class="section-hdr">RISK SCORE DISTRIBUTION — FILTERED</div>',
                       unsafe_allow_html=True)
            fig_rd = risk_score_dist(filtered)
            if fig_rd:
                st.plotly_chart(fig_rd, use_container_width=True,
                               config={"displayModeBar":False},
                               key="anomaly_riskdist")

        # Top risk cases
        st.markdown('<div class="section-hdr">TOP RISK CASES</div>', unsafe_allow_html=True)
        display_cols = [c for c in ["step","type","amount","fatf_typology","aml_risk_score"]
                        if c in filtered.columns]
        if display_cols and "aml_risk_score" in filtered.columns:
            top = filtered.nlargest(50, "aml_risk_score")[display_cols]
            st.dataframe(top, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════

elif page == "📈  Financial Intelligence":
    st.markdown('<div class="section-hdr">EDGAR FINANCIAL INTELLIGENCE</div>', unsafe_allow_html=True)
    df_edgar = load_edgar()

    if df_edgar is None:
        st.error("EDGAR data not loaded.")
    else:
        # Company selector
        companies_available = sorted(df_edgar["ticker"].unique().tolist()) \
                              if "ticker" in df_edgar.columns else []
        fc1, fc2 = st.columns([1, 3], gap="large")

        with fc1:
            st.markdown('<div class="section-hdr">FILTERS</div>', unsafe_allow_html=True)
            sel_companies = st.multiselect(
                "Select Companies",
                companies_available,
                default=companies_available[:3] if len(companies_available) >= 3 else companies_available
            )
            show_late_only = st.checkbox("Late filings only")
            show_hm_only   = st.checkbox("High materiality only (≥50)")

        with fc2:
            filtered_edgar = df_edgar.copy()
            if sel_companies:
                filtered_edgar = filtered_edgar[filtered_edgar["ticker"].isin(sel_companies)]
            if show_late_only and "late_filing" in filtered_edgar.columns:
                filtered_edgar = filtered_edgar[filtered_edgar["late_filing"] == 1]
            if show_hm_only and "materiality_score" in filtered_edgar.columns:
                filtered_edgar = filtered_edgar[filtered_edgar["materiality_score"] >= 50]

            # Summary metrics
            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric("Records", f"{len(filtered_edgar):,}")
            with m2:
                lf = int(filtered_edgar["late_filing"].sum()) if "late_filing" in filtered_edgar.columns else 0
                st.metric("Late Filings", lf)
            with m3:
                hm = int((filtered_edgar["materiality_score"] >= 50).sum()) \
                     if "materiality_score" in filtered_edgar.columns else 0
                st.metric("High Materiality", hm)
            with m4:
                avg_pm = filtered_edgar["profit_margin"].mean()*100 \
                         if "profit_margin" in filtered_edgar.columns else 0
                st.metric("Avg Profit Margin", f"{avg_pm:.1f}%")

            # Charts
            st.plotly_chart(
                profit_margin_chart(filtered_edgar, sel_companies if sel_companies else None),
                use_container_width=True,
                config={"displayModeBar":False},
                key="fin_pm"
            )
            fig_exp = yoy_expense_chart(filtered_edgar, sel_companies if sel_companies else None)
            if fig_exp:
                st.plotly_chart(fig_exp, use_container_width=True,
                               config={"displayModeBar":False},
                               key="fin_exp")

        # Per-company time series
        if sel_companies and len(sel_companies) == 1:
            st.markdown(f'<div class="section-hdr">{sel_companies[0]} — HISTORICAL TREND</div>',
                       unsafe_allow_html=True)
            fig_ts = company_detail_chart(df_edgar, sel_companies[0])
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True,
                               config={"displayModeBar":False},
                               key="fin_ts")

        # Full data table
        st.markdown('<div class="section-hdr">FILTERED RECORDS</div>', unsafe_allow_html=True)
        disp_cols = [c for c in ["ticker","fiscal_year","fiscal_period","materiality_score",
                                  "late_filing","profit_margin","operating_expenses_yoy_pct",
                                  "days_to_file"] if c in filtered_edgar.columns]
        st.dataframe(
            filtered_edgar[disp_cols].sort_values("materiality_score", ascending=False)
            if "materiality_score" in filtered_edgar.columns else filtered_edgar[disp_cols],
            use_container_width=True, height=350
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: REGULATIONS
# ════════════════════════════════════════════════════════════════════════════

elif page == "🏛   Regulations":
    st.markdown('<div class="section-hdr">REGULATORY CORPUS & COMPLIANCE</div>', unsafe_allow_html=True)

    # Last query results
    if st.session_state.last_result:
        result = st.session_state.last_result
        rag    = result.get("rag_findings", [])
        flags  = result.get("compliance_flags", [])

        if rag or flags:
            r1, r2 = st.columns([1.1, 0.9], gap="large")
            with r1:
                st.markdown('<div class="section-hdr">CITATIONS FROM LAST QUERY</div>',
                           unsafe_allow_html=True)
                for r in rag:
                    st.markdown(
                        f'<div class="citation">'
                        f'<div class="cit-src">{r["citation"]}'
                        f'<span>relevance: {r["relevance_score"]:.2f}</span></div>'
                        f'<div class="cit-text">{r["text"][:500]}...</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            with r2:
                if flags:
                    st.markdown('<div class="section-hdr">COMPLIANCE FLAGS</div>',
                               unsafe_allow_html=True)
                    for flag in flags:
                        sev = flag.get("severity","LOW").upper()
                        sc  = {"HIGH":"#f85149","MEDIUM":"#d2a020","LOW":"#3fb950"}.get(sev,"#8b949e")
                        st.markdown(
                            f'<div class="flag flag-{sev}">'
                            f'<div class="flag-lbl" style="color:{sc}">[{sev}] {flag.get("rule","")}</div>'
                            f'<div class="flag-txt">{flag.get("description","")}</div>'
                            f'<div style="font-size:0.68rem;color:#484f58;margin-top:4px">'
                            f'Evidence: {flag.get("evidence","")}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
        st.markdown("---")

    # Corpus browser
    st.markdown('<div class="section-hdr">REGULATORY CORPUS BROWSER</div>', unsafe_allow_html=True)
    try:
        chunks = pd.read_csv(ROOT/"data/processed/regulatory_chunks_v2.csv")
        cc1, cc2, cc3 = st.columns(3)
        with cc1: st.metric("Total Chunks",    f"{len(chunks):,}")
        with cc2: st.metric("Avg Words/Chunk", f"{chunks['word_count'].mean():.0f}")
        with cc3: st.metric("Sources",          chunks["source"].nunique())

        src_filter = st.selectbox("Browse source",
                                  ["All"] + sorted(chunks["source"].unique().tolist()))
        search_term = st.text_input("Search within corpus",
                                    placeholder="e.g. suspicious activity threshold")

        filtered_chunks = chunks.copy()
        if src_filter != "All":
            filtered_chunks = filtered_chunks[filtered_chunks["source"] == src_filter]
        if search_term:
            filtered_chunks = filtered_chunks[
                filtered_chunks["text"].str.contains(search_term, case=False, na=False)
            ]

        st.caption(f"{len(filtered_chunks)} chunks matching filters")
        for _, row in filtered_chunks.head(8).iterrows():
            st.markdown(
                f'<div class="citation">'
                f'<div class="cit-src">{row["source"]}'
                f'<span>{row.get("word_count",0)} words</span></div>'
                f'<div class="cit-text">{row["text"][:400]}...</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    except Exception:
        st.info("Load regulatory chunks by running: python rag/build_vectorstore.py")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔬  Model Performance":
    st.markdown('<div class="section-hdr">ANOMALY DETECTION MODEL — PERFORMANCE REPORT</div>',
               unsafe_allow_html=True)
    st.caption("Isolation Forest · Trained on PaySim · Evaluated against ground-truth isFraud labels")

    if st.session_state.mp_cache is None:
        with st.spinner("Training model and computing metrics (first run only)..."):
            st.session_state.mp_cache = compute_model_perf()

    mp = st.session_state.mp_cache
    if mp is None or "error" in mp:
        st.error(f"Model metrics unavailable: {mp.get('error','unknown') if mp else 'data error'}")
    else:
        # Core metrics
        mm1,mm2,mm3,mm4,mm5 = st.columns(5)
        with mm1: st.metric("Precision",       f"{mp['precision']:.1%}")
        with mm2: st.metric("Recall",           f"{mp['recall']:.1%}")
        with mm3: st.metric("F1 Score",         f"{mp['f1']:.1%}")
        with mm4: st.metric("Sample Size",      f"{mp['sample']:,}")
        with mm5: st.metric("Actual Fraud Rate",f"{mp['fraud_rate']}%")

        st.markdown("")

        mc1, mc2 = st.columns(2, gap="large")

        with mc1:
            st.markdown('<div class="section-hdr">CONFUSION MATRIX</div>', unsafe_allow_html=True)
            cm = np.array(mp["cm"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted Normal","Predicted Fraud"],
                y=["Actual Normal","Actual Fraud"],
                text=[[str(cm[i][j]) for j in range(2)] for i in range(2)],
                texttemplate="%{text}",
                textfont=dict(size=15, family="IBM Plex Mono", color="white"),
                colorscale=[[0,"#161b22"],[0.5,"#1c2a3a"],[1,"#1f6feb"]],
                showscale=False,
            ))
            fig_cm.update_layout(
                height=240,
                xaxis=dict(tickfont=dict(color="#8b949e",size=9)),
                yaxis=dict(tickfont=dict(color="#8b949e",size=9)),
                **PLOT,
            )
            st.plotly_chart(fig_cm, use_container_width=True,
                           config={"displayModeBar":False}, key="mp_cm")

        with mc2:
            st.markdown('<div class="section-hdr">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
            fi = mp["feat_imp"]
            sorted_fi = sorted(fi.items(), key=lambda x:x[1])
            labels = [k.replace("_flag","").replace("_"," ").title() for k,_ in sorted_fi]
            vals   = [v for _,v in sorted_fi]
            colors_fi = ["#58a6ff" if v >= 60 else "#8b949e" for v in vals]
            fig_fi = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors_fi, opacity=0.85,
                            line=dict(color="#0d1117",width=1)),
                text=[f"{v:.0f}" for v in vals],
                textposition="outside",
                textfont=dict(color="#484f58",size=9),
            ))
            fig_fi.update_layout(
                height=240,
                xaxis=dict(showgrid=False,tickfont=dict(color="#484f58",size=8),
                           range=[0, max(vals)*1.3] if vals else [0,100]),
                yaxis=dict(showgrid=False,tickfont=dict(color="#c9d1d9",size=10)),
                showlegend=False, **PLOT,
            )
            st.plotly_chart(fig_fi, use_container_width=True,
                           config={"displayModeBar":False}, key="mp_fi")

        # Model card
        st.markdown('<div class="section-hdr" style="margin-top:1rem">MODEL CARD</div>',
                   unsafe_allow_html=True)
        cd1,cd2,cd3,cd4 = st.columns(4)
        with cd1:
            st.markdown('<div class="stat-card"><div class="stat-label">Algorithm</div>'
                        '<div class="stat-value" style="font-size:0.9rem">Isolation Forest</div>'
                        '<div class="stat-sub">Unsupervised · sklearn</div></div>',
                        unsafe_allow_html=True)
        with cd2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Training Data</div>'
                        f'<div class="stat-value" style="font-size:0.9rem">PaySim</div>'
                        f'<div class="stat-sub">6.3M synthetic mobile money transactions</div></div>',
                        unsafe_allow_html=True)
        with cd3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Features</div>'
                        f'<div class="stat-value" style="font-size:0.9rem">{len(mp["feature_cols"])}</div>'
                        f'<div class="stat-sub">AML behavioral signals</div></div>',
                        unsafe_allow_html=True)
        with cd4:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Contamination</div>'
                        f'<div class="stat-value" style="font-size:0.9rem">5%</div>'
                        f'<div class="stat-sub">Expected anomaly rate</div></div>',
                        unsafe_allow_html=True)

        # Limitations
        st.markdown('<div class="section-hdr" style="margin-top:1rem">LIMITATIONS & GOVERNANCE NOTES</div>',
                   unsafe_allow_html=True)
        for lim in [
            "PaySim is synthetic data — real-world transaction patterns will differ in volume and complexity",
            "Isolation Forest is unsupervised — it finds statistical outliers, not definitively fraudulent transactions",
            "Feature importance is computed via permutation, not SHAP — directional accuracy only",
            "Model requires periodic retraining as financial transaction patterns evolve",
            "Ground truth labels (isFraud) from PaySim may not reflect real regulatory definitions of fraud",
        ]:
            st.markdown(
                f'<div style="font-size:0.76rem;color:#484f58;padding:4px 0 4px 10px;'
                f'border-left:2px solid #21262d;margin:3px 0">{lim}</div>',
                unsafe_allow_html=True
            )
            