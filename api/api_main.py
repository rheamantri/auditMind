"""
AuditMind v2 — FastAPI Backend
Wraps all 9 agents and serves structured data to the Lovable React frontend.
Run with: uvicorn api.main:app --reload --port 8000
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
PROCESSED = ROOT / "data/processed"

from agents.auditMind_agents import build_graph, AuditState

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(title="AuditMind API", version="2.0")
graph = build_graph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════════
# HELPER — Extract entity and analysis type from query
# ════════════════════════════════════════════════════════════════════════════

COMPANY_MAP = {
    "jpmorgan": "JPM", "jpm": "JPM",
    "bank of america": "BAC", "bac": "BAC",
    "goldman": "GS", "goldman sachs": "GS", "gs": "GS",
    "morgan stanley": "MS", "ms": "MS",
    "wells fargo": "WFC", "wfc": "WFC",
    "citigroup": "C", "citi": "C",
    "american express": "AXP", "amex": "AXP",
    "blackrock": "BLK", "blk": "BLK",
    "schwab": "SCHW", "charles schwab": "SCHW",
    "us bancorp": "USB", "usb": "USB",
}

def extract_entities(query: str) -> dict:
    q = query.lower()
    companies = [ticker for name, ticker in COMPANY_MAP.items() if name in q]
    return {
        "companies":     list(set(companies)),
        "is_comparison": any(w in q for w in ["compare", "vs", "versus", "against"]),
        "is_deep_dive":  len(companies) == 1,
        "focus_aml":     any(w in q for w in ["structuring", "aml", "layering", "smurfing",
                                                "fan-out", "transaction", "suspicious"]),
        "focus_financial": any(w in q for w in ["financial", "revenue", "profit", "expense",
                                                  "filing", "edgar", "balance sheet", "income"]),
        "focus_compliance": any(w in q for w in ["compliance", "fatf", "ffiec", "regulation",
                                                   "violation", "rule", "bsa"]),
        "focus_scenario": any(w in q for w in ["scenario", "risk", "impact", "ignore", "consequence"]),
    }


# ════════════════════════════════════════════════════════════════════════════
# HELPER — Build all chart data from processed CSVs
# ════════════════════════════════════════════════════════════════════════════

def build_aml_charts(entities: dict) -> dict:
    """Builds all AML chart data from PaySim"""
    try:
        df = pd.read_csv(PROCESSED / "paysim_high_risk.csv")

        # 1. Typology breakdown
        typology = {}
        if "fatf_typology" in df.columns:
            typology = df["fatf_typology"].value_counts().to_dict()

        # 2. Amount distribution — THE STRUCTURING CLIFF CHART
        # Bins from $8000-$11000 to show the pile-up below $10k
        if "amount" in df.columns:
            mask = (df["amount"] >= 8000) & (df["amount"] <= 11000)
            amount_data = df[mask]["amount"]
            hist_counts, hist_edges = np.histogram(amount_data, bins=60)
            amount_distribution = [
                {
                    "range": f"${hist_edges[i]:,.0f}",
                    "midpoint": round((hist_edges[i] + hist_edges[i+1]) / 2, 0),
                    "count": int(hist_counts[i]),
                    "is_threshold": hist_edges[i] <= 10000 <= hist_edges[i+1]
                }
                for i in range(len(hist_counts))
            ]
        else:
            amount_distribution = []

        # 3. Risk score distribution
        risk_distribution = []
        if "aml_risk_score" in df.columns:
            hist_counts, hist_edges = np.histogram(df["aml_risk_score"].dropna(), bins=20)
            risk_distribution = [
                {"score_range": f"{hist_edges[i]:.0f}-{hist_edges[i+1]:.0f}",
                 "count": int(hist_counts[i])}
                for i in range(len(hist_counts))
            ]

        # 4. Top 10 highest risk transactions
        top_risk = []
        if "aml_risk_score" in df.columns:
            cols = [c for c in ["step", "type", "amount", "fatf_typology", "aml_risk_score"]
                    if c in df.columns]
            top_risk = df.nlargest(10, "aml_risk_score")[cols].to_dict("records")

        # 5. FATF flag radar data (6 typology flags)
        radar_data = {}
        flag_cols = ["structuring_flag", "balance_drain", "balance_mismatch",
                     "funds_not_received", "high_velocity", "fan_out_flag"]
        for col in flag_cols:
            if col in df.columns:
                radar_data[col.replace("_flag", "").replace("_", " ").title()] = int(df[col].sum())

        # 6. Transaction volume by step (time proxy)
        volume_over_time = []
        if "step" in df.columns:
            vol = df.groupby("step").size().reset_index(name="count")
            vol = vol[vol["step"] <= 100]  # First 100 steps for readability
            volume_over_time = vol.to_dict("records")

        # Summary stats
        structuring_count = int(df["structuring_flag"].sum()) if "structuring_flag" in df.columns else 0
        avg_amount = float(df[df.get("structuring_flag", pd.Series([0]*len(df))) == 1]["amount"].mean()) \
                     if structuring_count > 0 and "amount" in df.columns else 0

        return {
            "typology_breakdown":    typology,
            "amount_distribution":   amount_distribution,
            "risk_distribution":     risk_distribution,
            "top_risk_transactions": top_risk,
            "radar_data":            radar_data,
            "volume_over_time":      volume_over_time,
            "stats": {
                "total_high_risk":        len(df),
                "structuring_count":      structuring_count,
                "avg_structuring_amount": round(avg_amount, 2),
                "unique_typologies":      len(typology),
            }
        }
    except Exception as e:
        return {"error": str(e), "stats": {}}


def build_financial_charts(entities: dict) -> dict:
    """Builds all EDGAR financial chart data"""
    try:
        df = pd.read_csv(PROCESSED / "edgar_processed.csv")
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

        companies = entities.get("companies", [])
        if companies:
            df_filtered = df[df["ticker"].isin(companies)]
        else:
            df_filtered = df

        # 1. Revenue trend — multi-line per company, last 12 quarters
        revenue_trend = []
        if "revenue" in df.columns and "ticker" in df.columns:
            rev_df = df[df["metric"] == "revenue"] if "metric" in df.columns else df
            if "revenue" in df.columns:
                rev_df = df[["ticker", "period_end", "revenue"]].dropna()
                rev_df = rev_df.sort_values("period_end")
                for ticker in rev_df["ticker"].unique():
                    company_data = rev_df[rev_df["ticker"] == ticker].tail(12)
                    for _, row in company_data.iterrows():
                        revenue_trend.append({
                            "ticker":  ticker,
                            "date":    str(row["period_end"])[:10] if pd.notna(row["period_end"]) else "",
                            "revenue": round(float(row["revenue"]) / 1e9, 2)  # In billions
                        })

        # 2. Profit margin comparison
        profit_margins = []
        if "profit_margin" in df.columns:
            latest = df.sort_values("period_end").groupby("ticker").last().reset_index()
            latest = latest[latest["profit_margin"].notna()]
            profit_margins = [
                {
                    "ticker": row["ticker"],
                    "margin": round(float(row["profit_margin"]) * 100, 2),
                    "is_negative": float(row["profit_margin"]) < 0
                }
                for _, row in latest.iterrows()
            ]

        # 3. Filing timeline — days to file vs deadline (40 days for 10-Q, 60 for 10-K)
        filing_timeline = []
        if "days_to_file" in df.columns and "ticker" in df.columns:
            late_df = df[df["days_to_file"].notna()].copy()
            late_df = late_df.sort_values("period_end").groupby("ticker").last().reset_index()
            for _, row in late_df.iterrows():
                days = float(row["days_to_file"])
                deadline = 60 if str(row.get("form_type", "")).startswith("10-K") else 40
                filing_timeline.append({
                    "ticker":        row["ticker"],
                    "days_to_file":  round(days, 0),
                    "deadline":      deadline,
                    "days_over":     max(0, round(days - deadline, 0)),
                    "is_late":       bool(row.get("late_filing", days > deadline))
                })

        # 4. Expense YoY change
        expense_changes = []
        if "operating_expenses_yoy_pct" in df.columns:
            latest = df.sort_values("period_end").groupby("ticker").last().reset_index()
            latest = latest[latest["operating_expenses_yoy_pct"].notna()]
            expense_changes = [
                {
                    "ticker": row["ticker"],
                    "yoy_pct": round(float(row["operating_expenses_yoy_pct"]), 2),
                    "is_spike": abs(float(row["operating_expenses_yoy_pct"])) > 20
                }
                for _, row in latest.iterrows()
            ]

        # 5. Risk heatmap — companies × risk dimensions
        risk_heatmap = []
        tickers = df["ticker"].unique() if "ticker" in df.columns else []
        for ticker in tickers:
            company_df = df[df["ticker"] == ticker]
            latest = company_df.sort_values("period_end").iloc[-1] if len(company_df) > 0 else None
            if latest is not None:
                risk_heatmap.append({
                    "ticker": ticker,
                    "aml_exposure":      50,  # Base — overridden by AML agent findings
                    "late_filing_risk":  min(100, int(latest.get("days_to_file", 0) or 0)),
                    "expense_anomaly":   min(100, abs(float(latest.get("operating_expenses_yoy_pct", 0) or 0)) * 2),
                    "profitability_risk": max(0, int((1 - float(latest.get("profit_margin", 0.5) or 0.5)) * 100)),
                    "materiality_score": int(latest.get("materiality_score", 0) or 0),
                })

        # 6. Peer comparison for single company deep dive
        peer_comparison = []
        if len(companies) == 1 and "profit_margin" in df.columns:
            ticker = companies[0]
            latest_all = df.sort_values("period_end").groupby("ticker").last().reset_index()
            metrics = ["profit_margin", "debt_to_equity", "cash_ratio"]
            for metric in metrics:
                if metric in latest_all.columns:
                    company_val = latest_all[latest_all["ticker"] == ticker][metric].values
                    peer_median = latest_all[metric].median()
                    if len(company_val) > 0:
                        peer_comparison.append({
                            "metric":       metric.replace("_", " ").title(),
                            "company":      round(float(company_val[0]), 4) if pd.notna(company_val[0]) else 0,
                            "peer_median":  round(float(peer_median), 4) if pd.notna(peer_median) else 0,
                        })

        # Summary stats
        late_count = int(df["late_filing"].sum()) if "late_filing" in df.columns else 0
        high_mat = int((df["materiality_score"] >= 50).sum()) if "materiality_score" in df.columns else 0

        return {
            "revenue_trend":      revenue_trend,
            "profit_margins":     profit_margins,
            "filing_timeline":    filing_timeline,
            "expense_changes":    expense_changes,
            "risk_heatmap":       risk_heatmap,
            "peer_comparison":    peer_comparison,
            "stats": {
                "companies_analyzed":   int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
                "late_filings_count":   late_count,
                "high_materiality":     high_mat,
                "total_records":        len(df),
            }
        }
    except Exception as e:
        return {"error": str(e), "stats": {}}


# ════════════════════════════════════════════════════════════════════════════
# MAIN ANALYZE ENDPOINT
# ════════════════════════════════════════════════════════════════════════════

@app.post("/analyze")
def analyze(request: QueryRequest):
    """
    Main endpoint. Runs all 9 agents + builds chart data.
    Returns fully structured response for the React frontend.
    """
    try:
        # 1. Run agent pipeline
        initial_state: AuditState = {
            "user_query":          request.query,
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
        result = graph.invoke(initial_state)

        # 2. Extract entities from query for dynamic chart filtering
        entities = extract_entities(request.query)

        # 3. Build rich chart data
        aml_charts      = build_aml_charts(entities)
        financial_charts = build_financial_charts(entities)

        # 4. Build decision rationale cards
        # Connect specific findings → specific regulations → specific recommendation
        compliance_flags = result.get("compliance_flags", [])
        rag_findings     = result.get("rag_findings", [])
        anomaly          = result.get("anomaly_findings", {})
        financial        = result.get("financial_findings", {})
        mat              = result.get("materiality_detail", {})

        rationale_cards = []
        if anomaly.get("data_available"):
            findings = anomaly.get("findings", {})
            struct   = findings.get("structuring_transactions", 0)
            if struct > 0:
                top_reg = rag_findings[0]["citation"] if rag_findings else "FATF Recommendation 10"
                rationale_cards.append({
                    "finding":    f"{struct:,} structuring transactions detected (avg ${findings.get('avg_structuring_amount',0):,.0f})",
                    "regulation": top_reg,
                    "implication": "Threshold avoidance pattern requires SAR filing within 30 days",
                    "severity":   "HIGH"
                })
        if compliance_flags:
            high_flags = [f for f in compliance_flags if f.get("severity") == "HIGH"]
            if high_flags:
                rationale_cards.append({
                    "finding":    high_flags[0].get("description", ""),
                    "regulation": high_flags[0].get("rule", ""),
                    "implication": high_flags[0].get("evidence", ""),
                    "severity":   "HIGH"
                })
        if financial.get("data_available"):
            fin_summary = financial.get("summary", {})
            late = fin_summary.get("late_filings_count", 0)
            if late > 0:
                rationale_cards.append({
                    "finding":    f"{late} late SEC filings detected",
                    "regulation": "SEC Rule 13a-13 (Quarterly Report Filing Deadlines)",
                    "implication": "Late filings trigger heightened scrutiny in next audit cycle",
                    "severity":   "MEDIUM"
                })

        # 5. Build scenario projection with numbers
        scenario_raw = result.get("scenario_projection", "")
        scenario_cards = [
            {
                "risk_type":   "Regulatory Risk",
                "color":       "#f87171",
                "description": "FinCEN penalties for AML violations range from $50,000 to $1.9B (HSBC precedent). Failure to file SARs: $500-$1M per violation under BSA.",
                "timeframe":   "30-90 days to enforcement action"
            },
            {
                "risk_type":   "Financial Risk",
                "color":       "#fbbf24",
                "description": "Remediation costs typically 3-5x the original fine. Increased compliance program costs estimated $2-10M annually post-enforcement.",
                "timeframe":   "6-18 months to full financial impact"
            },
            {
                "risk_type":   "Reputational Risk",
                "color":       "#a78bfa",
                "description": "Public enforcement actions reduce customer trust measurably. Peer institutions face 8-15% deposit outflows following AML enforcement.",
                "timeframe":   "Immediate upon public disclosure"
            }
        ]

        # If scenario projection has actual content, prepend it
        if scenario_raw and len(scenario_raw) > 50:
            scenario_cards[0]["ai_analysis"] = scenario_raw

        # 6. Confidence breakdown for waterfall chart
        explainability     = result.get("explainability", {})
        confidence_factors = [
            {"factor": "Baseline",               "value": 50, "cumulative": 50,  "type": "base"},
            {"factor": "Transaction data",        "value": 15, "cumulative": 65,  "type": "add"} if anomaly.get("data_available") else {"factor": "Transaction data", "value": 0, "cumulative": 50, "type": "neutral"},
            {"factor": "Financial data",          "value": 10, "cumulative": 75,  "type": "add"} if financial.get("data_available") else {"factor": "Financial data", "value": 0, "cumulative": 50, "type": "neutral"},
            {"factor": "Regulatory sources",      "value": 10, "cumulative": 85,  "type": "add"} if len(rag_findings) >= 2 else {"factor": "Regulatory sources", "value": 0, "cumulative": 50, "type": "neutral"},
            {"factor": "Compliance flags",        "value": 5,  "cumulative": 90,  "type": "add"} if compliance_flags else {"factor": "Compliance flags", "value": 0, "cumulative": 50, "type": "neutral"},
            {"factor": "Agent consistency",       "value": 8,  "cumulative": 98,  "type": "add"} if mat.get("level") == "HIGH" and compliance_flags else {"factor": "Agent consistency", "value": 0, "cumulative": 50, "type": "neutral"},
        ]

        # 7. Assemble complete response
        return {
            # ── Core agent outputs ──────────────────────────────────────
            "decision":            result.get("decision", "INVESTIGATE"),
            "decision_rationale":  result.get("decision_rationale", ""),
            "rationale_cards":     rationale_cards,
            "materiality_detail":  mat,
            "confidence_score":    result.get("confidence_score", 0.5),
            "confidence_pct":      f"{int(result.get('confidence_score', 0.5) * 100)}%",
            "confidence_factors":  confidence_factors,
            "query_intent":        result.get("query_intent", "mixed"),
            "agents_executed":     result.get("agents_executed", []),
            "audit_trail":         result.get("audit_trail", []),

            # ── Findings ────────────────────────────────────────────────
            "anomaly_findings":   result.get("anomaly_findings", {}),
            "financial_findings": result.get("financial_findings", {}),
            "rag_findings":       result.get("rag_findings", []),
            "compliance_flags":   result.get("compliance_flags", []),

            # ── Scenario ────────────────────────────────────────────────
            "scenario_cards":     scenario_cards,
            "scenario_raw":       scenario_raw,

            # ── Chart data ──────────────────────────────────────────────
            "aml_charts":         aml_charts,
            "financial_charts":   financial_charts,

            # ── Query context ───────────────────────────────────────────
            "entities":           entities,
            "query":              request.query,

            # ── Explainability ──────────────────────────────────────────
            "explainability":     explainability,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════
# STATIC DATA ENDPOINTS — for base charts that don't need agent pipeline
# ════════════════════════════════════════════════════════════════════════════

@app.get("/data/risk-heatmap")
def get_risk_heatmap():
    """Returns company × risk dimension heatmap data"""
    try:
        financial_charts = build_financial_charts({"companies": []})
        return {"heatmap": financial_charts.get("risk_heatmap", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/aml-summary")
def get_aml_summary():
    """Returns AML summary stats for base dashboard"""
    try:
        aml = build_aml_charts({"companies": []})
        return {
            "stats":             aml.get("stats", {}),
            "typology_breakdown": aml.get("typology_breakdown", {}),
            "radar_data":         aml.get("radar_data", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/financial-summary")
def get_financial_summary():
    """Returns financial summary for base dashboard"""
    try:
        fin = build_financial_charts({"companies": []})
        return {
            "stats":          fin.get("stats", {}),
            "profit_margins": fin.get("profit_margins", []),
            "filing_timeline": fin.get("filing_timeline", []),
            "risk_heatmap":   fin.get("risk_heatmap", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status":  "online",
        "version": "2.0",
        "agents":  9,
        "data_layers": ["PaySim", "EDGAR", "FATF/FFIEC"]
    }
