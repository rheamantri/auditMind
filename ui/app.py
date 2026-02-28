"""
AuditMind — Streamlit UI
Conversational financial audit intelligence platform.
"""

import sys
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.auditMind_agents import build_graph, AuditState

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AuditMind",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* ── Header bar ── */
.audit-header {
    background: linear-gradient(135deg, #0f1729 0%, #1a2744 50%, #0f1729 100%);
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.audit-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #60a5fa;
    letter-spacing: 0.15em;
    margin: 0;
}
.audit-subtitle {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
.status-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 8px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Decision badge ── */
.decision-badge {
    display: inline-block;
    padding: 0.6rem 1.4rem;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    margin: 0.5rem 0;
}
.badge-ESCALATE   { background:#4c1d1d; color:#f87171; border:1px solid #f87171; }
.badge-INVESTIGATE{ background:#1d2f4c; color:#60a5fa; border:1px solid #60a5fa; }
.badge-REVIEW     { background:#2d2a1a; color:#fbbf24; border:1px solid #fbbf24; }
.badge-DISCLOSE   { background:#1d3a2d; color:#34d399; border:1px solid #34d399; }
.badge-IGNORE     { background:#1e1e2e; color:#94a3b8; border:1px solid #4b5563; }

/* ── Metric card ── */
.metric-card {
    background: #0f1729;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.metric-label {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #60a5fa;
}

/* ── Chat message ── */
.chat-user {
    background: #1a2744;
    border: 1px solid #2d4a7a;
    border-radius: 10px 10px 2px 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.9rem;
}
.chat-assistant {
    background: #0f1729;
    border: 1px solid #1e3a5f;
    border-radius: 10px 10px 10px 2px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ── Agent trace ── */
.agent-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.4rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    border-left: 2px solid #1e3a5f;
    padding-left: 0.8rem;
    margin-left: 0.4rem;
}
.agent-step.done {
    color: #34d399;
    border-left-color: #34d399;
}
.agent-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #34d399;
    flex-shrink: 0;
}

/* ── Citation card ── */
.citation-card {
    background: #0a1628;
    border-left: 3px solid #3b82f6;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.78rem;
}
.citation-source {
    color: #3b82f6;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.05em;
}
.citation-score {
    float: right;
    color: #64748b;
    font-size: 0.68rem;
}

/* ── Flag row ── */
.flag-high   { border-left: 3px solid #f87171; background: #1a0f0f; }
.flag-medium { border-left: 3px solid #fbbf24; background: #1a1600; }
.flag-low    { border-left: 3px solid #34d399; background: #0f1a14; }
.flag-row {
    padding: 0.5rem 0.8rem;
    border-radius: 0 6px 6px 0;
    margin: 0.3rem 0;
    font-size: 0.78rem;
}

/* ── Demo query button ── */
.stButton > button {
    background: #0f1729 !important;
    color: #60a5fa !important;
    border: 1px solid #2d4a7a !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    padding: 0.4rem 0.8rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1a2744 !important;
    border-color: #60a5fa !important;
}

/* ── Input ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0f1729 !important;
    border: 1px solid #2d4a7a !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 8px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #060c18 !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #60a5fa !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0f1729 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2d4a7a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

if "messages"      not in st.session_state: st.session_state.messages      = []
if "last_result"   not in st.session_state: st.session_state.last_result   = None
if "query_count"   not in st.session_state: st.session_state.query_count   = 0
if "graph"         not in st.session_state:
    with st.spinner("Initializing agents..."):
        st.session_state.graph = build_graph()

DEMO_QUERIES = [
    "Which transactions show structuring behavior consistent with FATF threshold avoidance?",
    "Analyze JPMorgan's financial statements for materiality anomalies and late filings",
    "What AML compliance gaps exist based on FFIEC internal controls standards?",
    "Show me fan-out and layering patterns in the transaction data",
    "Compare Goldman Sachs vs Morgan Stanley operating expense trends",
]


# ════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def build_materiality_gauge(score: float) -> go.Figure:
    color = "#f87171" if score >= 70 else "#fbbf24" if score >= 40 else "#34d399"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": color, "family": "IBM Plex Mono", "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2d4a7a",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#0f1729",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#0d1f0d"},
                {"range": [40, 70], "color": "#1a1a0d"},
                {"range": [70,100], "color": "#1a0d0d"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": score
            }
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="#0a0e1a", font={"color": "#e2e8f0"},
    )
    return fig


def build_typology_chart(typology: dict) -> go.Figure:
    if not typology:
        return None
    labels = [k.split("(")[0].strip() for k in typology.keys()]
    values = list(typology.values())
    colors = ["#f87171","#fbbf24","#60a5fa","#34d399","#a78bfa","#fb923c"]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=colors[:len(labels)],
            line=dict(color="#0a0e1a", width=1)
        ),
        text=[f"{v:,}" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=10, family="IBM Plex Mono")
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1729",
        xaxis=dict(showgrid=False, color="#2d4a7a", tickfont=dict(color="#64748b", size=9)),
        yaxis=dict(showgrid=False, color="#2d4a7a", tickfont=dict(color="#94a3b8", size=10)),
        showlegend=False,
    )
    return fig


def build_confidence_bar(confidence: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[confidence * 100],
        y=["Confidence"],
        orientation="h",
        marker=dict(
            color="#3b82f6",
            line=dict(color="#0a0e1a", width=0)
        ),
        width=0.4,
    ))
    fig.add_shape(type="rect",
        x0=0, x1=100, y0=-0.5, y1=0.5,
        fillcolor="#0f1729", line=dict(color="#1e3a5f", width=1)
    )
    fig.update_layout(
        height=70, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        xaxis=dict(range=[0, 100], showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        showlegend=False,
        barmode="overlay"
    )
    return fig


def build_edgar_chart(result: dict) -> go.Figure | None:
    """Build a chart from EDGAR financial findings if available"""
    try:
        fin = result.get("financial_findings", {})
        if not fin.get("data_available"):
            return None

        df = pd.read_csv(ROOT / "data/processed/edgar_processed.csv")
        if "profit_margin" not in df.columns or "ticker" not in df.columns:
            return None

        # Latest profit margin per company
        latest = df.sort_values("period_end").groupby("ticker").last().reset_index()
        latest = latest[latest["profit_margin"].notna()].nlargest(8, "profit_margin")

        colors = ["#34d399" if v >= 0 else "#f87171" for v in latest["profit_margin"]]

        fig = go.Figure(go.Bar(
            x=latest["ticker"],
            y=(latest["profit_margin"] * 100).round(2),
            marker=dict(color=colors, line=dict(color="#0a0e1a", width=1)),
            text=[f"{v:.1f}%" for v in (latest["profit_margin"] * 100)],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=10)
        ))
        fig.update_layout(
            title=dict(text="Profit Margin by Company (Latest Quarter)",
                      font=dict(color="#64748b", size=11, family="IBM Plex Sans")),
            height=240,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0f1729",
            xaxis=dict(showgrid=False, color="#2d4a7a",
                      tickfont=dict(color="#94a3b8", size=10,
                                   family="IBM Plex Mono")),
            yaxis=dict(showgrid=True, gridcolor="#1e3a5f",
                      color="#2d4a7a",
                      tickfont=dict(color="#64748b", size=9)),
            showlegend=False,
        )
        return fig
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# RUN AGENT PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str) -> dict:
    initial_state: AuditState = {
        "user_query":          query,
        "query_intent":        "",
        "agents_to_run":       [],
        "anomaly_findings":    {},
        "financial_findings":  {},
        "rag_findings":        [],
        "compliance_flags":    [],
        "materiality_score":   0.0,
        "materiality_detail":  {},
        "explainability":      {},
        "decision":            "",
        "decision_rationale":  "",
        "scenario_projection": "",
        "final_response":      "",
        "agents_executed":     [],
        "confidence_score":    0.0,
        "audit_trail":         [],
    }
    return st.session_state.graph.invoke(initial_state)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚖ AuditMind")
    st.markdown("---")

    st.markdown("**DEMO QUERIES**")
    st.caption("Click to run a preset query")
    for q in DEMO_QUERIES:
        if st.button(q[:55] + "..." if len(q) > 55 else q, key=f"demo_{q[:20]}"):
            st.session_state.pending_query = q

    st.markdown("---")

    # Show governance panel if there's a result
    if st.session_state.last_result:
        result = st.session_state.last_result

        st.markdown("**GOVERNANCE PANEL**")

        # Agents executed
        agents = result.get("agents_executed", [])
        st.caption("Agents executed")
        for agent in agents:
            st.markdown(
                f'<div class="agent-step done">'
                f'<div class="agent-dot"></div>{agent}</div>',
                unsafe_allow_html=True
            )

        st.markdown("")

        # Confidence
        conf = result.get("confidence_score", 0)
        st.caption(f"AI Confidence: {int(conf*100)}%")
        st.plotly_chart(
            build_confidence_bar(conf),
            use_container_width=True,
            config={"displayModeBar": False}
        )

        # Data sources
        exp = result.get("explainability", {})
        sources = exp.get("data_sources_used", [])
        if sources:
            st.caption("Data sources")
            for s in sources:
                st.markdown(f"<span style='color:#34d399;font-size:0.72rem'>✓</span> "
                           f"<span style='font-size:0.72rem'>{s}</span>",
                           unsafe_allow_html=True)

        # Limitations
        limits = exp.get("limitations", [])
        if limits:
            with st.expander("⚠ Limitations", expanded=False):
                for l in limits:
                    st.markdown(f"<span style='font-size:0.72rem;color:#64748b'>{l}</span>",
                               unsafe_allow_html=True)

        # Audit trail
        trail = result.get("audit_trail", [])
        if trail:
            with st.expander("🔍 Audit Trail", expanded=False):
                for entry in trail:
                    st.markdown(
                        f"<div style='font-size:0.68rem;font-family:IBM Plex Mono;"
                        f"color:#64748b;padding:2px 0'>"
                        f"<span style='color:#3b82f6'>{entry.get('agent','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.65rem;color:#2d4a7a'>"
        "AuditMind v1.0 | Stack: LangGraph + Groq + ChromaDB + SEC EDGAR"
        "</span>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="audit-header">
  <div>
    <div class="audit-title">AUDITM<span style="color:#3b82f6">I</span>ND</div>
    <div class="audit-subtitle">Agentic AI · Financial Audit · AML Intelligence</div>
  </div>
  <div style="text-align:right;font-size:0.72rem;color:#2d4a7a;font-family:'IBM Plex Mono',monospace">
    <div><span class="status-dot"></span>SYSTEM ONLINE</div>
    <div style="margin-top:4px">9 AGENTS · 3 DATA LAYERS · FATF/FFIEC CORPUS</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Two-column layout
col_chat, col_analysis = st.columns([1.1, 0.9], gap="medium")

# ── LEFT: Chat ────────────────────────────────────────────────────────────────
with col_chat:
    st.markdown("#### Query Interface")

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🔍 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Input
    query = st.text_input(
        "Enter your audit query",
        placeholder="e.g. Which transactions show structuring behavior?",
        label_visibility="collapsed",
        key="query_input"
    )

    col_submit, col_clear = st.columns([3, 1])
    with col_submit:
        submit = st.button("▶ Run Analysis", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.last_result = None
            st.rerun()

    # Handle demo query click
    if "pending_query" in st.session_state:
        query  = st.session_state.pending_query
        submit = True
        del st.session_state.pending_query

    # Run pipeline
    if submit and query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.query_count += 1

        with st.spinner("Running 9-agent analysis pipeline..."):
            result = run_pipeline(query)

        st.session_state.last_result = result

        # Format response for chat
        decision = result.get("decision", "UNKNOWN")
        mat      = result.get("materiality_detail", {})
        rationale = result.get("decision_rationale", "")

        response_text = (
            f"**Decision:** `{decision}`\n\n"
            f"**Materiality:** {mat.get('score', 0)}/100 — {mat.get('level', '')}\n\n"
            f"**Rationale:** {rationale}\n\n"
            f"*See the Analysis Panel → for full breakdown, charts, citations and scenario projection.*"
        )
        level_color = {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#34d399"}.get(
            mat.get("level", "LOW"), "#94a3b8"
        )
        dec_color = {"ESCALATE": "#f87171", "INVESTIGATE": "#60a5fa",
                    "REVIEW": "#fbbf24", "DISCLOSE": "#34d399",
                    "IGNORE": "#94a3b8"}.get(decision, "#94a3b8")

        response_html = (
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:1rem;"
            f"font-weight:600;color:{dec_color}'>{decision}</span>"
            f"<br><br>"
            f"<span style='color:#64748b;font-size:0.75rem;text-transform:uppercase;"
            f"letter-spacing:0.08em'>Materiality</span>&nbsp;&nbsp;"
            f"<span style='font-family:IBM Plex Mono,monospace;color:{level_color}'>"
            f"{mat.get('score', 0)}/100 — {mat.get('level', '')}</span>"
            f"<br><br>"
            f"<span style='color:#94a3b8'>{rationale}</span>"
            f"<br><br>"
            f"<span style='color:#4b5563;font-size:0.8rem;font-style:italic'>"
            f"→ See Anomalies, Regulations, Financials and Scenario tabs for full breakdown.</span>"
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_html
        })


# ── RIGHT: Analysis Panel ─────────────────────────────────────────────────────
with col_analysis:
    if st.session_state.last_result:
        result = st.session_state.last_result

        # ── Decision badge ──
        decision = result.get("decision", "UNKNOWN")
        st.markdown(
            f'<div class="decision-badge badge-{decision}">{decision}</div>',
            unsafe_allow_html=True
        )
        st.caption(result.get("decision_rationale", "")[:200])

        # ── Tabs ──
        intent = result.get("query_intent", "mixed")

        # Label the active tab so user knows what's most relevant
        tab_labels = {
            "aml":        ["📊 Anomalies ●", "🏛 Regulations", "📈 Financials", "🎯 Scenario"],
            "financial":  ["📈 Financials ●", "📊 Anomalies", "🏛 Regulations", "🎯 Scenario"],
            "compliance": ["🏛 Regulations ●", "📊 Anomalies", "📈 Financials", "🎯 Scenario"],
        }.get(intent, ["📊 Anomalies", "🏛 Regulations", "📈 Financials", "🎯 Scenario"])

        tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

        with tab1:
            mat = result.get("materiality_detail", {})
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Materiality Score**")
                fig = build_materiality_gauge(mat.get("score", 0))
                st.plotly_chart(fig, use_container_width=True,
                               config={"displayModeBar": False})

            with c2:
                st.markdown("**Risk Level**")
                level = mat.get("level", "LOW")
                color = {"HIGH":"#f87171","MEDIUM":"#fbbf24","LOW":"#34d399"}.get(level,"#94a3b8")
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Overall Risk</div>'
                    f'<div class="metric-value" style="color:{color}">{level}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                factors = mat.get("factors", [])
                for f in factors[:3]:
                    st.markdown(
                        f"<div style='font-size:0.72rem;color:#64748b;"
                        f"padding:2px 0;border-left:2px solid #1e3a5f;"
                        f"padding-left:8px;margin:3px 0'>{f}</div>",
                        unsafe_allow_html=True
                    )

            # Typology chart
            anomaly = result.get("anomaly_findings", {})
            if anomaly.get("data_available"):
                findings = anomaly.get("findings", {})
                typology = findings.get("typology_breakdown", {})
                if typology:
                    st.markdown("**AML Typology Breakdown**")
                    fig2 = build_typology_chart(typology)
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True,
                                       config={"displayModeBar": False})

                # Key metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Structuring Tx",
                             f"{findings.get('structuring_transactions',0):,}")
                with m2:
                    st.metric("High-Risk Tx",
                             f"{findings.get('total_high_risk_transactions',0):,}")
                with m3:
                    avg = findings.get("avg_structuring_amount", 0)
                    st.metric("Avg Amount", f"${avg:,.0f}")

                st.markdown("**AML Analysis**")
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#94a3b8;line-height:1.7'>"
                    f"{anomaly.get('interpretation','')}</div>",
                    unsafe_allow_html=True
                )

        with tab2:
            rag = result.get("rag_findings", [])
            if rag:
                st.markdown("**Regulatory Citations**")
                for r in rag:
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<span class="citation-score">relevance: {r["relevance_score"]:.2f}</span>'
                        f'<div class="citation-source">{r["citation"]}</div>'
                        f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;'
                        f'line-height:1.6">{r["text"][:300]}...</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                #st.info("No regulatory citations retrieved for this query.")
                # In tab2 (Regulations) — when no RAG results
                st.markdown(
                    "<div style='background:#0f1729;border:1px solid #1e3a5f;border-radius:8px;"
                    "padding:1.2rem;text-align:center;margin-top:1rem'>"
                    "<div style='color:#4b5563;font-size:0.8rem;margin-bottom:0.6rem'>"
                    "No regulatory citations for this query type.</div>"
                    "<div style='color:#3b82f6;font-size:0.75rem;font-style:italic'>"
                    "Try: \"What does FFIEC say about suspicious activity reporting?\"</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

            flags = result.get("compliance_flags", [])
            if flags:
                st.markdown("**Compliance Flags**")
                for flag in flags:
                    sev = flag.get("severity", "LOW").upper()
                    st.markdown(
                        f'<div class="flag-row flag-{sev.lower()}">'
                        f'<strong style="font-size:0.72rem">[{sev}] {flag.get("rule","")}</strong><br>'
                        f'<span style="font-size:0.76rem;color:#94a3b8">'
                        f'{flag.get("description","")}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with tab3:
            fin = result.get("financial_findings", {})
            if fin.get("data_available"):
                summary = fin.get("summary", {})
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Companies", summary.get("companies_analyzed", 0))
                with m2:
                    st.metric("Late Filings", summary.get("late_filings_count", 0))
                with m3:
                    st.metric("High Materiality", summary.get("high_materiality_count", 0))

                fig3 = build_edgar_chart(result)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True,
                                   config={"displayModeBar": False})

                st.markdown("**Financial Analysis**")
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#94a3b8;line-height:1.7'>"
                    f"{fin.get('interpretation','')}</div>",
                    unsafe_allow_html=True
                )
            else:
                #st.info("Run a financial query to see EDGAR statement analysis here.")
                #st.markdown(
                #    "_Try: 'Analyze JPMorgan financial statements for materiality anomalies'_"
                #)
                # In tab3 (Financials) — when no financial data
                st.markdown(
                    "<div style='background:#0f1729;border:1px solid #1e3a5f;border-radius:8px;"
                    "padding:1.2rem;text-align:center;margin-top:1rem'>"
                    "<div style='color:#4b5563;font-size:0.8rem;margin-bottom:0.6rem'>"
                    "No financial statement analysis for this query.</div>"
                    "<div style='color:#3b82f6;font-size:0.75rem;font-style:italic'>"
                    "Try: \"Analyze JPMorgan financial statements for materiality anomalies\"</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

        with tab4:
            scenario = result.get("scenario_projection", "")
            if scenario:
                st.markdown("**Risk Scenario: If Findings Are Ignored**")
                lines = scenario.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("*") or line.startswith("•") or line.startswith("-"):
                        line = line.lstrip("*•- ").strip()
                        if ":" in line:
                            label, rest = line.split(":", 1)
                            risk_color = (
                                "#f87171" if "regulatory" in label.lower()
                                else "#fbbf24" if "financial" in label.lower()
                                else "#a78bfa"
                            )
                            st.markdown(
                                f'<div style="background:#0f1729;border-left:3px solid {risk_color};'
                                f'padding:0.7rem 1rem;margin:0.4rem 0;border-radius:0 6px 6px 0">'
                                f'<div style="color:{risk_color};font-size:0.7rem;'
                                f'font-family:IBM Plex Mono,monospace;text-transform:uppercase;'
                                f'letter-spacing:0.08em;margin-bottom:4px">{label.strip()}</div>'
                                f'<div style="font-size:0.8rem;color:#94a3b8;line-height:1.6">'
                                f'{rest.strip()}</div></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div style="font-size:0.82rem;color:#94a3b8;'
                                f'padding:4px 0">{line}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#94a3b8;'
                            f'padding:4px 0">{line}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.info("Scenario projection will appear here after analysis.")

    else:
        # Empty state
        st.markdown(
            """
            <div style='text-align:center;padding:4rem 2rem;color:#2d4a7a'>
              <div style='font-size:3rem;margin-bottom:1rem'>⚖</div>
              <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;
                          letter-spacing:0.1em;color:#1e3a5f'>
                AWAITING QUERY
              </div>
              <div style='font-size:0.75rem;margin-top:0.5rem;color:#1a2744'>
                Type a query or select a demo from the sidebar
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )