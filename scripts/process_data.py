"""
AuditMind — Data Processing Pipeline
Cleans and engineers features for:
  1. EDGAR financial statements  → materiality + anomaly features
  2. PaySim transactions         → AML / structuring features
  3. BankSim transactions        → expense fraud features
  4. Regulatory documents        → chunked text for RAG
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
RAW_TX    = ROOT / "data/raw/transactions"
RAW_EDGAR = ROOT / "data/raw/edgar"
RAW_DOCS  = ROOT / "data/raw/regulatory_docs"
PROCESSED = ROOT / "data/processed"
PROCESSED.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. EDGAR — Financial Statement Processing
# ════════════════════════════════════════════════════════════════════════════

def process_edgar():
    print("\n" + "="*60)
    print("  Processing EDGAR Financial Data")
    print("="*60)

    csv_files = list(RAW_EDGAR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "_summary.csv"]

    if not csv_files:
        print("  ✗ No EDGAR CSVs found. Run fetch_data.py first.")
        return

    # Load and combine all companies
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} total records across {df['ticker'].nunique()} companies")

    # ── Clean dates ──────────────────────────────────────────────────────────
    df["period_end"]  = pd.to_datetime(df["period_end"],  errors="coerce")
    df["filed_date"]  = pd.to_datetime(df["filed_date"],  errors="coerce")
    df["period_start"]= pd.to_datetime(df["period_start"],errors="coerce")
    df = df.dropna(subset=["period_end", "value"])
    df = df[df["value"] != 0]

    # ── Keep only quarterly and annual filings ───────────────────────────────
    df = df[df["form_type"].isin(["10-K", "10-Q"])]

    # ── Pivot: one row per company+period, columns = metrics ─────────────────
    pivot = df.pivot_table(
        index=["ticker", "company", "fiscal_year", "fiscal_period",
               "period_end", "form_type", "filed_date"],
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()
    pivot.columns.name = None

    # ── Feature Engineering ──────────────────────────────────────────────────
    print("  Engineering financial features...")

    # Days to file — audit signal (SEC requires 10-Q within 40 days, 10-K within 60)
    pivot["days_to_file"] = (pivot["filed_date"] - pivot["period_end"]).dt.days

    # Late filing flag
    pivot["late_filing"] = (
        ((pivot["form_type"] == "10-Q") & (pivot["days_to_file"] > 40)) |
        ((pivot["form_type"] == "10-K") & (pivot["days_to_file"] > 60))
    ).astype(int)

    # Year-over-year change for key metrics
    pivot = pivot.sort_values(["ticker", "period_end"])
    for col in ["revenue", "net_income", "operating_expenses", "total_assets"]:
        if col in pivot.columns:
            pivot[f"{col}_yoy_pct"] = (
                pivot.groupby("ticker")[col]
                .pct_change(periods=4)  # 4 quarters = 1 year
                .round(4) * 100
            )

    # Profit margin
    if "net_income" in pivot.columns and "revenue" in pivot.columns:
        pivot["profit_margin"] = (pivot["net_income"] / pivot["revenue"]).round(4)

    # Debt to equity ratio
    if "long_term_debt" in pivot.columns and "stockholders_equity" in pivot.columns:
        pivot["debt_to_equity"] = (
            pivot["long_term_debt"] / pivot["stockholders_equity"].replace(0, np.nan)
        ).round(4)

    # Cash ratio
    if "cash" in pivot.columns and "total_liabilities" in pivot.columns:
        pivot["cash_ratio"] = (
            pivot["cash"] / pivot["total_liabilities"].replace(0, np.nan)
        ).round(4)

    # ── Materiality Score ────────────────────────────────────────────────────
    # Flags anomalies that are material relative to company size
    print("  Calculating materiality scores...")

    if "operating_expenses" in pivot.columns and "revenue" in pivot.columns:
        # Expense ratio
        pivot["expense_ratio"] = (
            pivot["operating_expenses"] / pivot["revenue"].replace(0, np.nan)
        ).round(4)

        # Z-score of expense ratio per company (how abnormal is this quarter?)
        pivot["expense_ratio_zscore"] = (
            pivot.groupby("ticker")["expense_ratio"]
            .transform(lambda x: (x - x.mean()) / x.std())
            .round(4)
        )

        # Materiality score: 0-100
        # High score = material anomaly worth investigating
        def materiality_score(row):
            score = 0
            # Large YoY expense swing
            if pd.notna(row.get("operating_expenses_yoy_pct")):
                if abs(row["operating_expenses_yoy_pct"]) > 20:
                    score += 30
                elif abs(row["operating_expenses_yoy_pct"]) > 10:
                    score += 15
            # Statistical outlier
            if pd.notna(row.get("expense_ratio_zscore")):
                if abs(row["expense_ratio_zscore"]) > 2.5:
                    score += 35
                elif abs(row["expense_ratio_zscore"]) > 1.5:
                    score += 20
            # Late filing
            if row.get("late_filing") == 1:
                score += 20
            # Negative profit margin
            if pd.notna(row.get("profit_margin")) and row["profit_margin"] < 0:
                score += 15
            return min(score, 100)

        pivot["materiality_score"] = pivot.apply(materiality_score, axis=1)

    # ── Industry peer comparison ─────────────────────────────────────────────
    if "profit_margin" in pivot.columns:
        pivot["profit_margin_vs_peers"] = (
            pivot.groupby(["fiscal_year", "fiscal_period"])["profit_margin"]
            .transform(lambda x: x - x.median())
            .round(4)
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = PROCESSED / "edgar_processed.csv"
    pivot.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(pivot):,} rows → {out_path}")

    # Save a high-materiality subset for quick agent access
    if "materiality_score" in pivot.columns:
        high_mat = pivot[pivot["materiality_score"] >= 50].copy()
        high_mat.to_csv(PROCESSED / "edgar_high_materiality.csv", index=False)
        print(f"  ✓ High materiality cases: {len(high_mat):,} rows → edgar_high_materiality.csv")

    return pivot


# ════════════════════════════════════════════════════════════════════════════
# 2. PaySim — AML Transaction Processing
# ════════════════════════════════════════════════════════════════════════════

def process_paysim():
    print("\n" + "="*60)
    print("  Processing PaySim Transaction Data")
    print("="*60)

    # Find the CSV
    paysim_files = list((RAW_TX / "paysim").glob("*.csv"))
    if not paysim_files:
        print("  ✗ No PaySim CSV found.")
        return

    print(f"  Loading {paysim_files[0].name}...")
    df = pd.read_csv(paysim_files[0])
    print(f"  Loaded {len(df):,} transactions")
    print(f"  Columns: {list(df.columns)}")

    # Standardize column names (PaySim uses camelCase)
    # Standardize column names (PaySim uses camelCase)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Fix column name inconsistency (Org vs Orig)
    df = df.rename(columns={
        "oldbalanceorg":  "oldbalanceorig",
        "newbalanceorg":  "newbalanceorig",
        "oldbalancedest": "oldbalancedest",
        "newbalancedest": "newbalancedest",
        "nameorig":       "nameorig",
        "namedest":       "namedest",
        "isfraud":        "isfraud",
        "isflaggedfraud": "isflaggedfraud"
    })
    print(f"  Columns after renaming: {list(df.columns)}")

    # ── Basic cleaning ───────────────────────────────────────────────────────
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0]

    # ── AML Feature Engineering ──────────────────────────────────────────────
    print("  Engineering AML features...")

    # 1. Structuring flag — transactions just below $10,000 reporting threshold
    df["structuring_flag"] = (
        (df["amount"] >= 9000) & (df["amount"] < 10000)
    ).astype(int)

    # 2. Balance drain flag — sender's account fully emptied
    df["balance_drain"] = (
        (df["newbalanceorig"] == 0) &
        (df["oldbalanceorig"] > 0)
    ).astype(int)

    # 3. Balance mismatch — expected vs actual balance after transaction
    df["expected_balance"] = df["oldbalanceorig"] - df["amount"]
    df["balance_mismatch"] = (
        abs(df["newbalanceorig"] - df["expected_balance"]) > 1
    ).astype(int)

    # 4. Funds not received — transfer sent but destination unchanged
    df["funds_not_received"] = (
        (df["type"] == "TRANSFER") &
        (df["oldbalancedest"] == df["newbalancedest"])
    ).astype(int)

    # 5. Transaction velocity — how many transactions does this sender make per hour?
    tx_velocity = df.groupby(["nameorig", "step"]).size().reset_index(name="tx_velocity")
    df = df.merge(tx_velocity, on=["nameorig", "step"], how="left")
    df["high_velocity"] = (df["tx_velocity"] > 3).astype(int)

    # 6. Fan-out — one sender to many unique destinations (layering pattern)
    fan_out = df.groupby("nameorig")["namedest"].nunique().reset_index(name="unique_destinations")
    df = df.merge(fan_out, on="nameorig", how="left")
    df["fan_out_flag"] = (df["unique_destinations"] > 5).astype(int)

    # 7. AML Risk Score — composite
    df["aml_risk_score"] = (
        df["structuring_flag"]   * 25 +
        df["balance_drain"]      * 20 +
        df["balance_mismatch"]   * 20 +
        df["funds_not_received"] * 20 +
        df["high_velocity"]      * 10 +
        df["fan_out_flag"]       * 15
    ).clip(0, 100)

    # 8. FATF typology label — what kind of AML pattern is this?
    def fatf_typology(row):
        if row["structuring_flag"] == 1:
            return "Structuring (FATF Rec. 10)"
        elif row["funds_not_received"] == 1 and row["balance_drain"] == 1:
            return "Layering (FATF Rec. 10/16)"
        elif row["fan_out_flag"] == 1:
            return "Fan-out / Smurfing (FATF Rec. 10)"
        elif row["high_velocity"] == 1:
            return "High Velocity (FATF Rec. 10)"
        elif row["balance_mismatch"] == 1:
            return "Balance Manipulation"
        else:
            return "Normal"

    df["fatf_typology"] = df.apply(fatf_typology, axis=1)

    # ── Save full dataset ────────────────────────────────────────────────────
    out_path = PROCESSED / "paysim_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} rows → {out_path}")

    # Save high-risk subset (agents don't need 6M rows at query time)
    high_risk = df[df["aml_risk_score"] >= 40].copy()
    high_risk.to_csv(PROCESSED / "paysim_high_risk.csv", index=False)
    print(f"  ✓ High-risk transactions: {len(high_risk):,} rows → paysim_high_risk.csv")

    # Save summary stats
    summary = {
        "total_transactions": len(df),
        "flagged_structuring": int(df["structuring_flag"].sum()),
        "flagged_balance_drain": int(df["balance_drain"].sum()),
        "flagged_funds_not_received": int(df["funds_not_received"].sum()),
        "high_risk_count": len(high_risk),
        "fraud_in_dataset": int(df.get("isfraud", pd.Series([0])).sum()),
        "typology_breakdown": df["fatf_typology"].value_counts().to_dict()
    }
    with open(PROCESSED / "paysim_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary stats → paysim_summary.json")

    return df


# ════════════════════════════════════════════════════════════════════════════
# 3. BankSim — Expense Fraud Processing
# ════════════════════════════════════════════════════════════════════════════

def process_banksim():
    print("\n" + "="*60)
    print("  Processing BankSim Transaction Data")
    print("="*60)

    banksim_dir = RAW_TX / "banksim"
    csv_files   = list(banksim_dir.glob("*.csv"))
    if not csv_files:
        print("  ✗ No BankSim CSV found.")
        return

    # Load the main transaction file (not the network file)
    main_file = [f for f in csv_files if "NET" not in f.name]
    if not main_file:
        main_file = csv_files
    
    print(f"  Loading {main_file[0].name}...")
    df = pd.read_csv(main_file[0], quotechar="'")
    print(f"  Loaded {len(df):,} transactions")
    print(f"  Columns: {list(df.columns)}")

    df.columns = [c.strip().lower().replace("'", "") for c in df.columns]

    # ── Clean ────────────────────────────────────────────────────────────────
    df = df.dropna(subset=["amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"] > 0]

    # ── Feature Engineering ──────────────────────────────────────────────────
    print("  Engineering expense fraud features...")

    # 1. Amount zscore per category — is this spend unusual for this category?
    df["amount_zscore"] = (
        df.groupby("category")["amount"]
        .transform(lambda x: (x - x.mean()) / x.std())
        .round(4)
    )

    # 2. High spend flag — more than 2 std devs above category mean
    df["high_spend_flag"] = (df["amount_zscore"] > 2).astype(int)

    # 3. Customer spend velocity — transactions per day per customer
    velocity = df.groupby(["customer", "step"]).size().reset_index(name="daily_tx_count")
    df = df.merge(velocity, on=["customer", "step"], how="left")
    df["unusual_velocity"] = (df["daily_tx_count"] > 5).astype(int)

    # 4. Category concentration — customer spending unusually in one category
    cat_spend = df.groupby(["customer", "category"])["amount"].sum().reset_index(name="cat_total")
    cust_total = df.groupby("customer")["amount"].sum().reset_index(name="cust_total")
    cat_spend = cat_spend.merge(cust_total, on="customer")
    cat_spend["category_concentration"] = (cat_spend["cat_total"] / cat_spend["cust_total"]).round(4)
    df = df.merge(cat_spend[["customer","category","category_concentration"]],
                  on=["customer","category"], how="left")

    # 5. Fraud risk score
    df["fraud_risk_score"] = (
        df["high_spend_flag"]    * 35 +
        df["unusual_velocity"]   * 25 +
        (df["amount_zscore"].clip(0, 4) / 4 * 40)
    ).clip(0, 100).round(1)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = PROCESSED / "banksim_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} rows → {out_path}")

    high_risk = df[df["fraud_risk_score"] >= 40].copy()
    high_risk.to_csv(PROCESSED / "banksim_high_risk.csv", index=False)
    print(f"  ✓ High-risk transactions: {len(high_risk):,} rows → banksim_high_risk.csv")

    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. REGULATORY DOCS — Chunk for RAG
# ════════════════════════════════════════════════════════════════════════════

def process_regulatory_docs():
    print("\n" + "="*60)
    print("  Processing Regulatory Documents for RAG")
    print("="*60)

    try:
        import fitz         # PyMuPDF — for PDFs
        from docx import Document  # python-docx — for Word files
    except ImportError:
        print("  Installing missing packages...")
        os.system("pip install PyMuPDF python-docx -q")
        import fitz
        from docx import Document

    chunks = []
    chunk_id = 0

    def clean_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text

    def chunk_text(text, source, chunk_size=500, overlap=50):
        """Split text into overlapping chunks for better RAG retrieval"""
        nonlocal chunk_id
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text_str = " ".join(chunk_words)
            if len(chunk_text_str) > 100:  # Skip tiny chunks
                result.append({
                    "chunk_id":  chunk_id,
                    "source":    source,
                    "text":      chunk_text_str,
                    "char_count": len(chunk_text_str)
                })
                chunk_id += 1
            i += chunk_size - overlap
        return result

    # Process PDFs
    for pdf_path in RAW_DOCS.glob("*.pdf"):
        print(f"  Processing PDF: {pdf_path.name}")
        try:
            doc = fitz.open(str(pdf_path))
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            full_text = clean_text(full_text)
            new_chunks = chunk_text(full_text, source=pdf_path.stem)
            chunks.extend(new_chunks)
            print(f"    → {len(new_chunks)} chunks from {len(doc)} pages")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Process DOCX files
    for docx_path in RAW_DOCS.glob("*.docx"):
        print(f"  Processing DOCX: {docx_path.name}")
        try:
            doc = Document(str(docx_path))
            full_text = " ".join([p.text for p in doc.paragraphs if p.text.strip()])
            full_text = clean_text(full_text)
            new_chunks = chunk_text(full_text, source=docx_path.stem)
            chunks.extend(new_chunks)
            print(f"    → {len(new_chunks)} chunks")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Save all chunks
    chunks_df = pd.DataFrame(chunks)
    out_path = PROCESSED / "regulatory_chunks.csv"
    chunks_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Total chunks: {len(chunks_df):,} → {out_path}")
    print(f"  Sources: {chunks_df['source'].value_counts().to_dict()}")

    return chunks_df


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  AuditMind — Data Processing Pipeline")
    print("█"*60)

    process_edgar()
    process_paysim()
    process_banksim()
    process_regulatory_docs()

    print("\n" + "█"*60)
    print("  ✓ Processing complete. Files in data/processed/:")
    for f in sorted(PROCESSED.glob("*")):
        size = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size:>8.0f} KB")
    print("█"*60 + "\n")