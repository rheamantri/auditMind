"""
AuditMind — Agent Orchestration Layer
Nine specialized agents wired together via LangGraph.

Flow:
  User Query
      │
  Planner Agent          ← decides which agents to run
      │
  ┌───┴──────────────┐
  Anomaly Agent    Data Analyst Agent    RAG Agent
  Compliance Agent
      │
  Materiality Engine
      │
  Explainability Agent
      │
  Decision Agent
      │
  Scenario Simulator
      │
  Final Response (with citations, score, recommendation)
"""

import os
import json
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data/processed"

# ── Add project root to path so imports work ─────────────────────────────────
sys.path.insert(0, str(ROOT))
from rag.build_vectorstore import query_regulations

# ── LangGraph + Anthropic ────────────────────────────────────────────────────
#from langgraph.graph import StateGraph, END
#import anthropic

#client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

#MODEL = "claude-sonnet-4-20250514"

# ── LangGraph ────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ── LLM: Groq primary, Gemini fallback ───────────────────────────────────────
from groq import Groq
#import google.generativeai as genai
from google import genai

groq_client = Groq(api_key=os.getenv("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"))
#genai.configure(api_key=os.getenv("AIzaSyBjI7o3bKbS5HBJUHs8azRV_IMSV3KsifM"))
#gemini_model = genai.GenerativeModel("gemini-1.5-flash")
gemini_client = genai.Client(api_key=os.getenv("AIzaSyBjI7o3bKbS5HBJUHs8azRV_IMSV3KsifM"))
GROQ_MODEL = "llama-3.3-70b-versatile"


# ════════════════════════════════════════════════════════════════════════════
# STATE — Shared memory passed between all agents
# ════════════════════════════════════════════════════════════════════════════

class AuditState(TypedDict):
    # Input
    user_query:          str

    # Planner output
    query_intent:        str       # "aml" | "financial" | "compliance" | "mixed"
    agents_to_run:       list[str]

    # Agent outputs
    anomaly_findings:    dict
    financial_findings:  dict
    rag_findings:        list[dict]
    compliance_flags:    list[dict]

    # Derived layers
    materiality_score:   float
    materiality_detail:  dict
    explainability:      dict

    # Final outputs
    decision:            str       # ESCALATE | INVESTIGATE | IGNORE | DISCLOSE | REVIEW
    decision_rationale:  str
    scenario_projection: str
    final_response:      str

    # Governance
    agents_executed:     list[str]
    confidence_score:    float
    audit_trail:         list[dict]


# ════════════════════════════════════════════════════════════════════════════
# HELPER — Call Claude with a structured prompt
# ════════════════════════════════════════════════════════════════════════════
'''
def call_claude(system: str, user: str, max_tokens: int = 1000) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return response.content[0].text


def call_claude_json(system: str, user: str, max_tokens: int = 1000) -> dict:
    """Calls Claude and parses JSON response"""
    system_with_json = system + "\n\nYou MUST respond with valid JSON only. No preamble, no explanation, no markdown backticks."
    raw = call_claude(system_with_json, user, max_tokens)
    # Strip any accidental markdown
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": "JSON parse failed", "raw": raw[:200]}
'''
def call_llm(system: str, user: str, max_tokens: int = 1000) -> str:
    """
    Calls Groq first (fast + free).
    Falls back to Gemini if Groq fails or rate limits.
    """
    # ── Try Groq first ───────────────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ]
        )
        return response.choices[0].message.content

    except Exception as groq_error:
        print(f"  [LLM] Groq failed ({groq_error.__class__.__name__}), falling back to Gemini...")

        # ── Gemini fallback ──────────────────────────────────────────────────
        try:
            prompt = f"{system}\n\n{user}"
            response = gemini_client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)
            return response.text

        except Exception as gemini_error:
            print(f"  [LLM] Gemini also failed: {gemini_error}")
            return f"LLM unavailable. Groq: {groq_error}. Gemini: {gemini_error}"


# Keep call_claude as an alias so no other code needs changing
call_claude = call_llm


def call_claude_json(system: str, user: str, max_tokens: int = 1000) -> dict:
    """Calls LLM and parses JSON response"""
    system_with_json = (
        system +
        "\n\nYou MUST respond with valid JSON only. "
        "No preamble, no explanation, no markdown backticks. "
        "Start your response with { and end with }"
    )
    raw = call_llm(system_with_json, user, max_tokens)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        #match = re.search(r'\{.*\}', raw, re.DOTALL)
        match = re.search(r"[{].*[}]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"error": "JSON parse failed", "raw": raw[:200]}
    
# ════════════════════════════════════════════════════════════════════════════
# AGENT 1 — PLANNER
# ════════════════════════════════════════════════════════════════════════════

def planner_agent(state: AuditState) -> AuditState:
    """
    Reads the user query and decides:
    - What the intent is (AML, financial analysis, compliance, mixed)
    - Which agents to invoke
    """
    print("\n[Planner Agent] Analyzing query...")

    result = call_claude_json(
        system="""You are a financial audit AI planner. 
        Classify the user's query and decide which specialized agents to invoke.
        
        Available agents:
        - anomaly_agent: For transaction-level fraud, AML, structuring detection
        - data_analyst_agent: For financial statement analysis, trends, ratios
        - rag_agent: For regulatory guidance, compliance questions, FATF/FFIEC
        - compliance_agent: For checking specific regulatory rules
        
        Return JSON with:
        {
          "query_intent": "aml" | "financial" | "compliance" | "mixed",
          "agents_to_run": ["agent1", "agent2"],
          "query_summary": "one sentence description of what user wants"
        }""",
        user=f"User query: {state['user_query']}"
    )

    state["query_intent"]   = result.get("query_intent", "mixed")
    state["agents_to_run"]  = result.get("agents_to_run",
        ["anomaly_agent", "data_analyst_agent", "rag_agent", "compliance_agent"])
    state["agents_executed"] = ["planner_agent"]
    state["audit_trail"]     = [{
        "agent": "planner_agent",
        "decision": f"Intent={state['query_intent']}, Running: {state['agents_to_run']}"
    }]

    print(f"  Intent: {state['query_intent']}")
    print(f"  Agents: {state['agents_to_run']}")
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 2 — DATA ANALYST
# ════════════════════════════════════════════════════════════════════════════

def data_analyst_agent(state: AuditState) -> AuditState:
    """
    Analyzes EDGAR financial statement data.
    Finds trends, peer comparisons, late filings, ratio anomalies.
    """
    if "data_analyst_agent" not in state.get("agents_to_run", []):
        state["financial_findings"] = {"skipped": True}
        return state

    print("\n[Data Analyst Agent] Analyzing EDGAR financial data...")

    try:
        df = pd.read_csv(PROCESSED / "edgar_processed.csv")

        # Get high-materiality cases
        high_mat = df[df.get("materiality_score", pd.Series([0]*len(df))) >= 50] \
                   if "materiality_score" in df.columns else pd.DataFrame()

        # Late filers
        late = df[df.get("late_filing", pd.Series([0]*len(df))) == 1] \
               if "late_filing" in df.columns else pd.DataFrame()

        # Worst profit margins
        if "profit_margin" in df.columns:
            worst_margins = df.nsmallest(5, "profit_margin")[
                ["ticker", "fiscal_year", "fiscal_period", "profit_margin"]
            ].to_dict("records")
        else:
            worst_margins = []

        # YoY expense spikes > 20%
        if "operating_expenses_yoy_pct" in df.columns:
            expense_spikes = df[df["operating_expenses_yoy_pct"].abs() > 20][
                ["ticker", "fiscal_year", "fiscal_period", "operating_expenses_yoy_pct"]
            ].head(10).to_dict("records")
        else:
            expense_spikes = []

        summary = {
            "total_records":        len(df),
            "companies_analyzed":   df["ticker"].nunique() if "ticker" in df.columns else 0,
            "high_materiality_count": len(high_mat),
            "late_filings_count":   len(late),
            "late_filers":          late["ticker"].unique().tolist() if len(late) > 0 and "ticker" in late.columns else [],
            "worst_profit_margins": worst_margins,
            "expense_spikes":       expense_spikes[:5],
        }

        # Ask Claude to interpret
        interpretation = call_claude(
            system="""You are a financial analyst. Summarize these audit findings concisely in 3-4 sentences. Focus on red flags.
            You are a financial audit AI planner. 
Classify the user's query and decide which specialized agents to invoke.

Available agents:
- anomaly_agent: For transaction-level fraud, AML, structuring detection
- data_analyst_agent: For financial statement analysis, trends, ratios
- rag_agent: For regulatory guidance, compliance questions, FATF/FFIEC
- compliance_agent: For checking specific regulatory rules

IMPORTANT: Always include rag_agent and compliance_agent in every query.
They provide regulatory grounding that is relevant to all audit questions.
""",
            user=f"Query: {state['user_query']}\n\nFinancial data summary: {json.dumps(summary, default=str)}"
        )

        state["financial_findings"] = {
            "summary":        summary,
            "interpretation": interpretation,
            "data_available": True
        }
        print(f"  Found {len(high_mat)} high-materiality cases, {len(late)} late filings")

    except Exception as e:
        state["financial_findings"] = {"error": str(e), "data_available": False}
        print(f"  Warning: {e}")

    state["agents_executed"].append("data_analyst_agent")
    state["audit_trail"].append({"agent": "data_analyst_agent", "status": "complete"})
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 3 — ANOMALY AGENT
# ════════════════════════════════════════════════════════════════════════════

def anomaly_agent(state: AuditState) -> AuditState:
    """
    Analyzes PaySim and BankSim transaction data.
    Flags AML patterns: structuring, layering, smurfing, high velocity.
    """
    if "anomaly_agent" not in state.get("agents_to_run", []):
        state["anomaly_findings"] = {"skipped": True}
        return state

    print("\n[Anomaly Agent] Scanning transaction data for AML patterns...")

    try:
        # Load high-risk subset (not full 6M rows)
        df = pd.read_csv(PROCESSED / "paysim_high_risk.csv")

        # Load summary stats
        with open(PROCESSED / "paysim_summary.json") as f:
            summary = json.load(f)

        # Top FATF typologies
        typology_counts = df["fatf_typology"].value_counts().to_dict() \
                          if "fatf_typology" in df.columns else {}

        # Highest risk transactions
        top_risk = df.nlargest(10, "aml_risk_score")[
            ["step", "type", "amount", "fatf_typology", "aml_risk_score"]
        ].to_dict("records") if "aml_risk_score" in df.columns else []

        # Structuring analysis
        structuring_count = int(df["structuring_flag"].sum()) \
                            if "structuring_flag" in df.columns else 0
        avg_structuring_amount = float(
            df[df["structuring_flag"] == 1]["amount"].mean()
        ) if structuring_count > 0 else 0

        findings = {
            "total_high_risk_transactions": len(df),
            "structuring_transactions":     structuring_count,
            "avg_structuring_amount":       round(avg_structuring_amount, 2),
            "typology_breakdown":           typology_counts,
            "top_risk_cases":               top_risk[:5],
            "overall_fraud_count":          summary.get("fraud_in_dataset", 0),
        }

        # Claude interprets
        interpretation = call_claude(
            system="""You are an AML analyst. Interpret these transaction findings.
            Reference specific FATF Recommendations where relevant.
            Be specific about risk levels and what action should be taken.
            Keep response to 4-5 sentences.""",
            user=f"Query: {state['user_query']}\n\nAML findings: {json.dumps(findings, default=str)}"
        )

        state["anomaly_findings"] = {
            "findings":       findings,
            "interpretation": interpretation,
            "data_available": True
        }
        print(f"  {structuring_count:,} structuring transactions flagged")
        print(f"  {len(df):,} high-risk transactions in scope")

    except Exception as e:
        state["anomaly_findings"] = {"error": str(e), "data_available": False}
        print(f"  Warning: {e}")

    state["agents_executed"].append("anomaly_agent")
    state["audit_trail"].append({"agent": "anomaly_agent", "status": "complete"})
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 4 — RAG DOCUMENT AGENT
# ════════════════════════════════════════════════════════════════════════════

def rag_agent(state: AuditState) -> AuditState:
    """
    Retrieves relevant regulatory passages from the vector store.
    Provides citations from FATF and FFIEC documents.
    """
    if "rag_agent" not in state.get("agents_to_run", []):
        state["rag_findings"] = []
        return state

    print("\n[RAG Agent] Retrieving regulatory guidance...")

    try:
        results = query_regulations(state["user_query"], n_results=3)
        state["rag_findings"] = results
        print(f"  Retrieved {len(results)} regulatory passages")
        for r in results:
            print(f"  [{r['relevance_score']:.3f}] {r['citation'][:60]}...")
    except Exception as e:
        state["rag_findings"] = []
        print(f"  Warning: {e}")

    state["agents_executed"].append("rag_agent")
    state["audit_trail"].append({
        "agent": "rag_agent",
        "citations": [r.get("citation", "") for r in state.get("rag_findings", [])]
    })
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 5 — COMPLIANCE AGENT
# ════════════════════════════════════════════════════════════════════════════

def compliance_agent(state: AuditState) -> AuditState:
    """
    Checks findings against specific regulatory rules.
    Generates compliance flags with severity levels.
    """
    # Always run compliance — flags are valuable regardless of query type
    # Only skip if planner explicitly set intent to "financial" with no regulatory angle
    if state.get("query_intent") == "financial" and "compliance_agent" not in state.get("agents_to_run", []):
        state["compliance_flags"] = []
        return state

    print("\n[Compliance Agent] Checking regulatory compliance...")

    # Build context from previous agents
    context = {
        "anomaly_findings":  state.get("anomaly_findings", {}),
        "financial_findings": state.get("financial_findings", {}),
        "rag_passages":      [r.get("text", "")[:300] for r in state.get("rag_findings", [])],
    }

    result = call_claude_json(
        system="""You are a compliance officer reviewing financial audit findings.
        Based on the findings, identify specific compliance violations or risks.
        
        Return JSON with this exact structure:
        {
          "flags": [
            {
              "rule": "FATF Recommendation 10",
              "description": "what the violation is",
              "severity": "HIGH" | "MEDIUM" | "LOW",
              "evidence": "what data supports this flag"
            }
          ],
          "overall_compliance_risk": "HIGH" | "MEDIUM" | "LOW"
        }""",
        user=f"Query: {state['user_query']}\n\nFindings context: {json.dumps(context, default=str)[:2000]}",
        max_tokens=1000
    )

    state["compliance_flags"] = result.get("flags", [])
    print(f"  {len(state['compliance_flags'])} compliance flags raised")
    for f in state["compliance_flags"]:
        print(f"  [{f.get('severity','?')}] {f.get('rule','?')}: {f.get('description','')[:60]}...")

    state["agents_executed"].append("compliance_agent")
    state["audit_trail"].append({
        "agent":    "compliance_agent",
        "flags":    len(state["compliance_flags"]),
        "risk":     result.get("overall_compliance_risk", "UNKNOWN")
    })
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 6 — MATERIALITY ENGINE
# ════════════════════════════════════════════════════════════════════════════

def materiality_engine(state: AuditState) -> AuditState:
    """
    Scores the combined findings by business materiality.
    Filters noise — only material findings proceed to Decision Agent.
    Score 0-100: 0=immaterial, 100=highly material
    """
    print("\n[Materiality Engine] Calculating materiality score...")

    score = 0
    factors = []

    # Factor 1: Number of high-risk AML transactions
    anomaly = state.get("anomaly_findings", {})
    if anomaly.get("data_available"):
        findings = anomaly.get("findings", {})
        structuring = findings.get("structuring_transactions", 0)
        high_risk   = findings.get("total_high_risk_transactions", 0)

        if structuring > 100000:
            score += 30
            factors.append(f"Very high structuring volume: {structuring:,} transactions")
        elif structuring > 10000:
            score += 20
            factors.append(f"Elevated structuring: {structuring:,} transactions")
        elif structuring > 1000:
            score += 10
            factors.append(f"Moderate structuring: {structuring:,} transactions")

        if high_risk > 1000000:
            score += 20
            factors.append(f"Large high-risk transaction pool: {high_risk:,}")

    # Factor 2: Compliance flags severity
    flags = state.get("compliance_flags", [])
    high_flags = [f for f in flags if f.get("severity") == "HIGH"]
    med_flags  = [f for f in flags if f.get("severity") == "MEDIUM"]
    score += len(high_flags) * 15
    score += len(med_flags)  * 7
    if high_flags:
        factors.append(f"{len(high_flags)} HIGH severity compliance flags")

    # Factor 3: Financial materiality from EDGAR
    financial = state.get("financial_findings", {})
    if financial.get("data_available"):
        fin_summary = financial.get("summary", {})
        if fin_summary.get("late_filings_count", 0) > 0:
            score += 10
            factors.append(f"{fin_summary['late_filings_count']} late SEC filings detected")
        if fin_summary.get("high_materiality_count", 0) > 0:
            score += 15
            factors.append(f"{fin_summary['high_materiality_count']} high-materiality financial cases")

    # Factor 4: RAG relevance — how many strong regulatory hits
    rag = state.get("rag_findings", [])
    strong_hits = [r for r in rag if r.get("relevance_score", 0) > 0.5]
    if len(strong_hits) >= 2:
        score += 10
        factors.append(f"{len(strong_hits)} highly relevant regulatory citations found")

    score = min(score, 100)

    state["materiality_score"]  = float(score)
    state["materiality_detail"] = {
        "score":   score,
        "factors": factors,
        "level":   "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW",
        "note":    "Material findings require escalation review" if score >= 70
                   else "Monitor for pattern development" if score >= 40
                   else "Below materiality threshold — routine monitoring"
    }

    print(f"  Materiality Score: {score}/100 ({state['materiality_detail']['level']})")
    for f in factors:
        print(f"  • {f}")

    state["agents_executed"].append("materiality_engine")
    state["audit_trail"].append({
        "agent": "materiality_engine",
        "score": score,
        "level": state["materiality_detail"]["level"]
    })
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 7 — EXPLAINABILITY AGENT
# ════════════════════════════════════════════════════════════════════════════

def explainability_agent(state: AuditState) -> AuditState:
    """
    Produces a full reasoning trace.
    Computes confidence score based on data quality and consistency.
    Makes the AI glass-box, not black-box.
    """
    print("\n[Explainability Agent] Building reasoning trace...")

    # Confidence factors
    confidence = 0.5  # baseline
    reasons    = []

    # Data availability boosts confidence
    if state.get("anomaly_findings", {}).get("data_available"):
        confidence += 0.15
        reasons.append("Transaction data analyzed")
    if state.get("financial_findings", {}).get("data_available"):
        confidence += 0.10
        reasons.append("Financial statement data analyzed")
    if len(state.get("rag_findings", [])) >= 2:
        confidence += 0.10
        reasons.append("Multiple regulatory sources retrieved")
    if len(state.get("compliance_flags", [])) > 0:
        confidence += 0.05
        reasons.append("Compliance flags corroborate findings")

    # Consistency check — do all agents agree on risk level?
    mat_level = state.get("materiality_detail", {}).get("level", "LOW")
    high_flags = [f for f in state.get("compliance_flags", [])
                  if f.get("severity") == "HIGH"]

    if mat_level == "HIGH" and high_flags:
        confidence += 0.10
        reasons.append("Materiality and compliance agents are consistent")
    elif mat_level == "LOW" and not high_flags:
        confidence += 0.05
        reasons.append("Both agents agree on low risk")

    confidence = min(confidence, 0.98)  # Never claim 100% confidence

    state["explainability"] = {
        "confidence_score": round(confidence, 2),
        "confidence_pct":   f"{round(confidence * 100)}%",
        "reasoning_chain":  reasons,
        "agents_that_ran":  state.get("agents_executed", []),
        "data_sources_used": [
            s for s in ["EDGAR XBRL", "PaySim Transactions", "BankSim", "FATF Corpus", "FFIEC Manual"]
            if (s == "EDGAR XBRL" and state.get("financial_findings", {}).get("data_available")) or
               (s == "PaySim Transactions" and state.get("anomaly_findings", {}).get("data_available")) or
               (s in ["FATF Corpus", "FFIEC Manual"] and len(state.get("rag_findings", [])) > 0)
        ],
        "limitations": [
            "PaySim is synthetic data — real transaction volumes may differ",
            "EDGAR data reflects reported figures, not underlying transactions",
            "Regulatory citations are based on semantic similarity, not legal interpretation"
        ]
    }
    state["confidence_score"] = confidence
    print(f"  Confidence: {state['explainability']['confidence_pct']}")

    state["agents_executed"].append("explainability_agent")
    state["audit_trail"].append({
        "agent":      "explainability_agent",
        "confidence": confidence
    })
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 8 — DECISION AGENT
# ════════════════════════════════════════════════════════════════════════════

def decision_agent(state: AuditState) -> AuditState:
    """
    Takes all findings and makes a structured recommendation.
    Outputs one of: ESCALATE | INVESTIGATE | REVIEW | DISCLOSE | IGNORE
    This is what turns analytics into advisory.
    """
    print("\n[Decision Agent] Generating recommendation...")

    # Compile full context for decision
    context = {
        "query":             state["user_query"],
        "materiality":       state.get("materiality_detail", {}),
        "compliance_flags":  state.get("compliance_flags", []),
        "rag_citations":     [
            {"source": r["citation"], "text": r["text"][:200]}
            for r in state.get("rag_findings", [])
        ],
        "anomaly_summary":   state.get("anomaly_findings", {}).get("interpretation", ""),
        "financial_summary": state.get("financial_findings", {}).get("interpretation", ""),
        "confidence":        state.get("confidence_score", 0.5),
    }

    result = call_claude_json(
        system="""You are a senior audit partner at a Big 4 firm making a final recommendation.
        
        Based on all findings, recommend ONE action:
        - ESCALATE: Immediate escalation to senior management / regulators required
        - INVESTIGATE: Deeper investigation warranted before conclusions
        - REVIEW: Schedule for formal review in next audit cycle  
        - DISCLOSE: Must be disclosed in financial statements or to regulators
        - IGNORE: Below materiality threshold, no action required
        
        Return JSON:
        {
          "decision": "ESCALATE" | "INVESTIGATE" | "REVIEW" | "DISCLOSE" | "IGNORE",
          "rationale": "2-3 sentence explanation citing specific findings and regulations",
          "priority": "IMMEDIATE" | "HIGH" | "MEDIUM" | "LOW",
          "regulatory_basis": "specific FATF/FFIEC rule that supports this decision"
        }""",
        user=json.dumps(context, default=str)[:3000],
        max_tokens=600
    )

    state["decision"]          = result.get("decision", "INVESTIGATE")
    state["decision_rationale"] = result.get("rationale", "")

    print(f"  Decision: {state['decision']} ({result.get('priority','?')} priority)")
    print(f"  Basis: {result.get('regulatory_basis','')[:80]}...")

    state["agents_executed"].append("decision_agent")
    state["audit_trail"].append({
        "agent":    "decision_agent",
        "decision": state["decision"],
        "priority": result.get("priority"),
        "basis":    result.get("regulatory_basis", "")
    })
    return state


# ════════════════════════════════════════════════════════════════════════════
# AGENT 9 — SCENARIO SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

def scenario_simulator(state: AuditState) -> AuditState:
    """
    Projects the risk impact if the findings are ignored.
    Turns detection into strategic advisory.
    """
    print("\n[Scenario Simulator] Projecting risk scenarios...")

    # Only simulate if findings are material
    if state.get("materiality_detail", {}).get("level") == "LOW":
        state["scenario_projection"] = (
            "Materiality score below threshold. "
            "No significant risk projection required. "
            "Continue routine monitoring."
        )
        state["agents_executed"].append("scenario_simulator")
        return state

    context = {
        "decision":          state.get("decision"),
        "materiality":       state.get("materiality_detail", {}),
        "anomaly_findings":  state.get("anomaly_findings", {}).get("interpretation", ""),
        "compliance_flags":  state.get("compliance_flags", []),
    }

    projection = call_claude(
        system="""You are a risk consultant projecting consequences.
        Answer: What happens if these findings are IGNORED?
        Be specific with estimated impact ranges where possible.
        Format as 3 bullet points covering: regulatory risk, financial risk, reputational risk.
        Keep each bullet to 1-2 sentences. Be direct and quantitative where possible.""",
        user=f"Findings summary: {json.dumps(context, default=str)[:2000]}",
        max_tokens=400
    )

    state["scenario_projection"] = projection
    print(f"  Scenario projection generated")

    state["agents_executed"].append("scenario_simulator")
    state["audit_trail"].append({"agent": "scenario_simulator", "status": "complete"})
    return state


# ════════════════════════════════════════════════════════════════════════════
# FINAL RESPONSE COMPOSER
# ════════════════════════════════════════════════════════════════════════════

def compose_response(state: AuditState) -> AuditState:
    """Assembles all agent outputs into a clean final response"""
    print("\n[Response Composer] Building final response...")

    rag = state.get("rag_findings", [])
    citations_text = "\n".join([
        f"  • {r['citation']} (relevance: {r['relevance_score']:.2f})"
        for r in rag
    ]) if rag else "  • No regulatory citations retrieved"

    flags = state.get("compliance_flags", [])
    flags_text = "\n".join([
        f"  • [{f.get('severity','?')}] {f.get('rule','?')}: {f.get('description','')}"
        for f in flags
    ]) if flags else "  • No compliance flags raised"

    explainability = state.get("explainability", {})
    mat            = state.get("materiality_detail", {})

    response = f"""
╔══════════════════════════════════════════════════════════════╗
  AUDITM\u0130ND ANALYSIS REPORT
  Query: {state['user_query']}
╚══════════════════════════════════════════════════════════════╝

━━━ DECISION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Recommendation:  {state.get('decision', 'UNKNOWN')}
  Rationale:       {state.get('decision_rationale', '')}

━━━ MATERIALITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Score:  {mat.get('score', 0)}/100  [{mat.get('level', 'UNKNOWN')}]
  Factors:
{chr(10).join(f'  • {f}' for f in mat.get('factors', []))}

━━━ TRANSACTION ANOMALIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {state.get('anomaly_findings', {}).get('interpretation', 'No transaction analysis performed')}

━━━ FINANCIAL STATEMENT FINDINGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {state.get('financial_findings', {}).get('interpretation', 'No financial analysis performed')}

━━━ COMPLIANCE FLAGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{flags_text}

━━━ REGULATORY CITATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{citations_text}

━━━ SCENARIO PROJECTION (IF IGNORED) ━━━━━━━━━━━━━━━━━━━━━━━━━
  {state.get('scenario_projection', 'N/A')}

━━━ AI GOVERNANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Confidence:     {explainability.get('confidence_pct', 'N/A')}
  Agents run:     {' → '.join(state.get('agents_executed', []))}
  Data sources:   {', '.join(explainability.get('data_sources_used', []))}
  Limitations:
{chr(10).join(f'  • {l}' for l in explainability.get('limitations', []))}
"""

    state["final_response"] = response
    return state


# ════════════════════════════════════════════════════════════════════════════
# WIRE THE GRAPH
# ════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AuditState)

    # Add all nodes
    graph.add_node("planner_agent",       planner_agent)
    graph.add_node("data_analyst_agent",  data_analyst_agent)
    graph.add_node("anomaly_agent",       anomaly_agent)
    graph.add_node("rag_agent",           rag_agent)
    graph.add_node("compliance_agent",    compliance_agent)
    graph.add_node("materiality_engine",  materiality_engine)
    graph.add_node("explainability_agent",explainability_agent)
    graph.add_node("decision_agent",      decision_agent)
    graph.add_node("scenario_simulator",  scenario_simulator)
    graph.add_node("compose_response",    compose_response)

    # Wire the flow
    graph.set_entry_point("planner_agent")
    graph.add_edge("planner_agent",        "data_analyst_agent")
    graph.add_edge("data_analyst_agent",   "anomaly_agent")
    graph.add_edge("anomaly_agent",        "rag_agent")
    graph.add_edge("rag_agent",            "compliance_agent")
    graph.add_edge("compliance_agent",     "materiality_engine")
    graph.add_edge("materiality_engine",   "explainability_agent")
    graph.add_edge("explainability_agent", "decision_agent")
    graph.add_edge("decision_agent",       "scenario_simulator")
    graph.add_edge("scenario_simulator",   "compose_response")
    graph.add_edge("compose_response",     END)

    return graph.compile()


# ════════════════════════════════════════════════════════════════════════════
# MAIN — Test the full pipeline
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  AuditMind — Agent Pipeline Test")
    print("█"*60)

    auditMind = build_graph()

    # Test query — this is the demo query
    test_query = (
        "Which transactions show structuring behavior consistent with "
        "FATF threshold avoidance, and what is the regulatory exposure "
        "for the financial institution?"
    )

    print(f"\nQuery: {test_query}\n")

    initial_state: AuditState = {
        "user_query":          test_query,
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

    result = auditMind.invoke(initial_state)
    print(result["final_response"])
