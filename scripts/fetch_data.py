"""
AuditMind Data Pipeline
Automatically fetches:
  1. PaySim transaction data (Kaggle)
  2. BankSim transaction data (Kaggle)
  3. EDGAR financial statements (SEC API)
  4. Regulatory PDFs (FATF / FFIEC)
"""

import os
import requests
import zipfile
import pandas as pd
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Path constants ───────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
RAW_TX        = ROOT / "data/raw/transactions"
RAW_EDGAR     = ROOT / "data/raw/edgar"
RAW_DOCS      = ROOT / "data/raw/regulatory_docs"
PROCESSED     = ROOT / "data/processed"

for p in [RAW_TX, RAW_EDGAR, RAW_DOCS, PROCESSED]:
    p.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "auditMind-project rhea.mantri@gmail.com"}  # SEC requires this


# ════════════════════════════════════════════════════════════════════════════
# 1.  TRANSACTION DATASETS  (Kaggle)
# ════════════════════════════════════════════════════════════════════════════

def download_kaggle_dataset(dataset_slug: str, dest_folder: Path, dataset_label: str):
    """
    Downloads and unzips a Kaggle dataset using the API token from .env
    dataset_slug format: 'owner/dataset-name'
    """
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise EnvironmentError("KAGGLE_API_TOKEN not found in .env file")

    print(f"\n{'='*60}")
    print(f"  Downloading: {dataset_label}")
    print(f"{'='*60}")

    dest_folder.mkdir(parents=True, exist_ok=True)
    zip_path = dest_folder / f"{dataset_label}.zip"

    # Kaggle API endpoint
    owner, name = dataset_slug.split("/")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{name}"

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True
    )

    if response.status_code != 200:
        print(f"  ✗ Failed to download {dataset_label}: HTTP {response.status_code}")
        print(f"    Response: {response.text[:200]}")
        return False

    # Stream download with progress bar
    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        desc=f"  {dataset_label}",
        total=total,
        unit="B",
        unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    # Unzip
    print(f"  Extracting {dataset_label}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_folder)

    zip_path.unlink()  # Remove zip after extraction
    print(f"  ✓ {dataset_label} saved to {dest_folder}")
    return True


def fetch_transaction_datasets():
    """Downloads PaySim and BankSim from Kaggle"""

    datasets = [
        {
            "slug":   "ealaxi/paysim1",
            "folder": RAW_TX / "paysim",
            "label":  "PaySim"
        },
        {
            "slug":   "ntnu-testimon-groups/banksim1",
            "folder": RAW_TX / "banksim",
            "label":  "BankSim"
        }
    ]

    for ds in datasets:
        success = download_kaggle_dataset(ds["slug"], ds["folder"], ds["label"])
        if not success:
            print(f"  ⚠ Skipping {ds['label']} — check your Kaggle token or dataset slug")


# ════════════════════════════════════════════════════════════════════════════
# 2.  EDGAR FINANCIAL DATA  (SEC free API)
# ════════════════════════════════════════════════════════════════════════════

# Major financial companies to pull — mix of banks, fintech, insurance
TARGET_COMPANIES = [
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "WFC",    # Wells Fargo
    "C",      # Citigroup
    "AXP",    # American Express
    "BLK",    # BlackRock
    "SCHW",   # Charles Schwab
    "USB",    # US Bancorp
]

# Financial metrics we want for each company
# These are the official XBRL tag names from the us-gaap taxonomy
FINANCIAL_TAGS = {
    # Income Statement
    "NetIncomeLoss":                        "net_income",
    "Revenues":                             "revenue",
    "OperatingExpenses":                    "operating_expenses",
    "GrossProfit":                          "gross_profit",
    "InterestExpense":                      "interest_expense",
    "IncomeTaxExpenseBenefit":              "tax_expense",

    # Balance Sheet
    "Assets":                               "total_assets",
    "Liabilities":                          "total_liabilities",
    "StockholdersEquity":                   "stockholders_equity",
    "CashAndCashEquivalentsAtCarryingValue":"cash",
    "LongTermDebt":                         "long_term_debt",

    # Cash Flow
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "NetCashProvidedByUsedInInvestingActivities": "investing_cash_flow",
    "NetCashProvidedByUsedInFinancingActivities": "financing_cash_flow",
}


def get_all_ciks() -> dict:
    """Fetches the master ticker→CIK mapping from SEC"""
    print("\n  Fetching SEC ticker list...")
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    tickers = r.json()

    # Flip to ticker→{cik, title} for easy lookup
    lookup = {}
    for entry in tickers.values():
        ticker = entry["ticker"].upper()
        cik    = str(entry["cik_str"]).zfill(10)  # SEC needs 10-digit zero-padded CIK
        lookup[ticker] = {"cik": cik, "title": entry["title"]}
    return lookup


def fetch_concept(cik: str, tag: str) -> list:
    """
    Pulls historical data for one financial metric from one company.
    Returns a list of quarterly/annual records.
    """
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=HEADERS)

    if r.status_code == 404:
        return []    # Company doesn't report this tag (common — not all tags apply to all companies)
    r.raise_for_status()

    data  = r.json()
    units = data.get("units", {})

    # Most financial values are in USD
    records = units.get("USD", units.get("shares", []))

    # Keep only records that have a 'frame' — these are deduplicated canonical records
    clean = [rec for rec in records if "frame" in rec]
    return clean


def fetch_edgar_data():
    """
    Pulls all financial metrics for all target companies.
    Saves one CSV per company.
    """
    print(f"\n{'='*60}")
    print(f"  Fetching EDGAR Financial Data")
    print(f"{'='*60}")

    cik_map = get_all_ciks()
    summary = []

    for ticker in tqdm(TARGET_COMPANIES, desc="  Companies"):
        if ticker not in cik_map:
            print(f"\n  ⚠ Ticker {ticker} not found in SEC database")
            continue

        info    = cik_map[ticker]
        cik     = info["cik"]
        company = info["title"]
        rows    = []

        for xbrl_tag, friendly_name in FINANCIAL_TAGS.items():
            records = fetch_concept(cik, xbrl_tag)
            for rec in records:
                rows.append({
                    "ticker":          ticker,
                    "company":         company,
                    "cik":             cik,
                    "metric":          friendly_name,
                    "xbrl_tag":        xbrl_tag,
                    "period_start":    rec.get("start", None),
                    "period_end":      rec["end"],
                    "value":           rec["val"],
                    "fiscal_year":     rec.get("fy", None),
                    "fiscal_period":   rec.get("fp", None),
                    "form_type":       rec.get("form", None),
                    "filed_date":      rec.get("filed", None),
                    "accession_number":rec.get("accn", None),
                    "frame":           rec.get("frame", None),
                })
            time.sleep(0.1)  # Respect SEC rate limits (10 req/sec max)

        if rows:
            df = pd.DataFrame(rows)

            # Add days_to_file — key audit signal (long delay = red flag)
            df["period_end"]  = pd.to_datetime(df["period_end"])
            df["filed_date"]  = pd.to_datetime(df["filed_date"])
            df["days_to_file"]= (df["filed_date"] - df["period_end"]).dt.days

            out_path = RAW_EDGAR / f"{ticker}.csv"
            df.to_csv(out_path, index=False)
            summary.append({"ticker": ticker, "company": company, "records": len(df)})
            print(f"\n  ✓ {ticker} ({company}): {len(df)} records saved")

    # Save a summary
    pd.DataFrame(summary).to_csv(RAW_EDGAR / "_summary.csv", index=False)
    print(f"\n  ✓ EDGAR data complete. Summary saved to data/raw/edgar/_summary.csv")


# ════════════════════════════════════════════════════════════════════════════
# 3.  REGULATORY DOCUMENTS  (FATF + FFIEC — public PDFs)
# ════════════════════════════════════════════════════════════════════════════

REGULATORY_DOCS = [
    {
        "name": "FATF_40_Recommendations",
        "url":  "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/FATF%20Recommendations%202012.pdf.coredownload.inline.pdf",
        "desc": "FATF 40 Recommendations — core AML/CFT standards"
    },
    {
        "name": "FATF_Risk_Based_Approach",
        "url":  "https://www.fatf-gafi.org/content/dam/fatf-gafi/guidance/RBA-Guidance-2007.pdf.coredownload.inline.pdf",
        "desc": "FATF Risk-Based Approach guidance"
    },
    {
        "name": "FFIEC_BSA_AML_Manual",
        "url":  "https://bsaaml.ffiec.gov/docs/manual/BSA_AML_Exam_Manual.pdf",
        "desc": "FFIEC BSA/AML Examination Manual — full"
    },
]


def fetch_regulatory_docs():
    """Downloads FATF and FFIEC PDFs for the RAG corpus"""
    print(f"\n{'='*60}")
    print(f"  Fetching Regulatory Documents")
    print(f"{'='*60}")

    for doc in REGULATORY_DOCS:
        out_path = RAW_DOCS / f"{doc['name']}.pdf"

        if out_path.exists():
            print(f"  ✓ Already exists: {doc['name']}")
            continue

        print(f"\n  Downloading: {doc['name']}")
        print(f"  Source: {doc['desc']}")

        try:
            r = requests.get(doc["url"], stream=True, timeout=30,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm(
                desc=f"  {doc['name']}",
                total=total,
                unit="B",
                unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

            print(f"  ✓ Saved: {out_path}")

        except Exception as e:
            print(f"  ✗ Failed to download {doc['name']}: {e}")
            print(f"    You can manually place the PDF in: {RAW_DOCS}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN — Run everything
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  AuditMind — Data Pipeline Starting")
    print("█"*60)

    print("\n[1/3] Transaction Datasets (Kaggle)...")
    fetch_transaction_datasets()

    print("\n[2/3] EDGAR Financial Data (SEC API)...")
    fetch_edgar_data()

    print("\n[3/3] Regulatory Documents (FATF / FFIEC)...")
    fetch_regulatory_docs()

    print("\n" + "█"*60)
    print("  ✓ All data fetched successfully")
    print(f"  Transactions  → {RAW_TX}")
    print(f"  EDGAR data    → {RAW_EDGAR}")
    print(f"  Regulatory    → {RAW_DOCS}")
    print("█"*60 + "\n")
'''

---

### 2B — Open `.env` and add your email (SEC requires it)

Open `.env` in VS Code and update it so it looks like this:
KAGGLE_API_TOKEN=KGAT_a298744829865972b7e18316e221fc34
ANTHROPIC_API_KEY=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
SEC_USER_AGENT=rhea.mantri@gmail.com
'''
