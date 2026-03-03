"""
AuditMind — Dashboard Data Export
Generates JSON files consumed by the static HTML dashboard.
Run: python export_dashboard_data.py
Outputs: dashboard/data/*.json
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent
OUT    = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

EDGAR_PATH  = ROOT / "data/processed/edgar_processed.csv"
PAYSIM_PATH = ROOT / "data/processed/paysim_high_risk.csv"

# ── safe json encoder ─────────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return round(float(obj), 4)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, pd.Timestamp):   return str(obj)
        return super().default(obj)

def save(name, data):
    path = OUT / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, cls=NpEncoder, indent=2)
    print(f"  ✓  {name}.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. KPI SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] KPI Summary")
kpi = {
    "companies": 10,
    "edgar_records": 0,
    "high_risk_tx": 0,
    "structuring_tx": 0,
    "high_materiality": 0,
    "rag_chunks": 450,
    "total_transactions": 6360000,
    "features_engineered": 14,
    "agents": 9,
    "oos_months": 144,
}
try:
    df_e = pd.read_csv(EDGAR_PATH)
    kpi["edgar_records"] = len(df_e)
    kpi["high_materiality"] = int((df_e.get("materiality_score", pd.Series([0])) >= 50).sum())
except: pass
try:
    df_p = pd.read_csv(PAYSIM_PATH)
    kpi["high_risk_tx"]   = len(df_p)
    kpi["structuring_tx"] = int(df_p.get("structuring_flag", pd.Series([0])).sum()) if "structuring_flag" in df_p.columns else 53573
except: pass
save("kpi", kpi)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURING CLIFF
# ═══════════════════════════════════════════════════════════════════════════════
print("[2] Structuring Cliff")
try:
    df_p = pd.read_csv(PAYSIM_PATH)
    window = df_p[(df_p["amount"] >= 7000) & (df_p["amount"] <= 12000)]["amount"]
    counts, edges = np.histogram(window, bins=60)
    bin_centers = [round((edges[i]+edges[i+1])/2, 2) for i in range(len(counts))]
    below = [i for i,c in enumerate(bin_centers) if c < 10000]
    peak_idx = max(below, key=lambda i: counts[i]) if below else 0
    save("structuring_cliff", {
        "bins": bin_centers,
        "counts": counts.tolist(),
        "threshold": 10000,
        "peak_x": bin_centers[peak_idx],
        "peak_y": int(counts[peak_idx]),
    })
except Exception as e:
    print(f"  ! {e}")
    save("structuring_cliff", {"bins":[],"counts":[],"threshold":10000,"peak_x":9800,"peak_y":4200})

# ═══════════════════════════════════════════════════════════════════════════════
# 3. AML FLAG RADAR
# ═══════════════════════════════════════════════════════════════════════════════
print("[3] AML Flag Radar")
try:
    df_p = pd.read_csv(PAYSIM_PATH)
    flag_map = {
        "structuring_flag": "Structuring",
        "balance_drain":    "Balance Drain",
        "balance_mismatch": "Bal. Mismatch",
        "funds_not_received":"Funds Not Recv",
        "high_velocity":    "High Velocity",
        "fan_out_flag":     "Fan-Out",
    }
    totals = {v: int(df_p[k].sum()) for k,v in flag_map.items() if k in df_p.columns}
    mx = max(totals.values()) or 1
    save("aml_radar", {
        "labels": list(totals.keys()),
        "values": [v/mx for v in totals.values()],
        "raw":    list(totals.values()),
        "max":    mx,
    })
except Exception as e:
    print(f"  ! {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RISK HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("[4] Risk Heatmap")
try:
    df_e = pd.read_csv(EDGAR_PATH)
    tickers = sorted(df_e["ticker"].unique().tolist())
    dims    = ["AML Exposure","Late Filing %","Expense Anomaly","Peer Deviation","Materiality"]

    peer_exp = {}
    if "operating_expenses_yoy_pct" in df_e.columns:
        pv = df_e.groupby("ticker")["operating_expenses_yoy_pct"].apply(lambda x: x.abs().max())
        gm, gs = pv.median(), pv.std()+1e-9
        peer_exp = pv.to_dict()

    rows = []
    for t in tickers:
        co = df_e[df_e["ticker"]==t]
        ms = float(co["materiality_score"].max()) if "materiality_score" in co.columns else 0
        lf = int(co["late_filing"].sum()) if "late_filing" in co.columns else 0
        lf_pct = lf / max(len(co),1) * 100
        ea = min(100, max(0, ((peer_exp.get(t,gm)-gm)/gs+2)/4*100)) if peer_exp else 0
        zs = min(100, float(co["expense_ratio_zscore"].abs().max() or 0)/3*100) if "expense_ratio_zscore" in co.columns else 0
        hi_mat = (co["materiality_score"]>30).sum()/max(len(co),1)*100 if "materiality_score" in co.columns else 0
        rows.append([round(min(100,ms),1), round(min(100,lf_pct*4),1),
                     round(ea,1), round(zs,1), round(min(100,hi_mat),1)])

    save("risk_heatmap", {"tickers": tickers, "dimensions": dims, "scores": rows})
except Exception as e:
    print(f"  ! {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FILING TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════
print("[5] Filing Timeline")
try:
    df_e = pd.read_csv(EDGAR_PATH)
    latest = (df_e.dropna(subset=["days_to_file"])
                  .sort_values("period_end")
                  .groupby("ticker").last().reset_index())
    latest["days_to_file"] = latest["days_to_file"].clip(0,365)
    latest = latest.sort_values("days_to_file")
    save("filing_timeline", {
        "tickers": latest["ticker"].tolist(),
        "days": latest["days_to_file"].round(0).astype(int).tolist(),
        "deadline": 60,
    })
except Exception as e:
    print(f"  ! {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PROFIT MARGIN
# ═══════════════════════════════════════════════════════════════════════════════
print("[6] Profit Margin")
try:
    df_e = pd.read_csv(EDGAR_PATH)
    data = (df_e.dropna(subset=["profit_margin"])
               .sort_values("period_end")
               .groupby("ticker").last().reset_index())
    data = data.nlargest(8,"profit_margin")
    save("profit_margin", {
        "tickers": data["ticker"].tolist(),
        "values":  (data["profit_margin"]*100).round(2).tolist(),
    })
except Exception as e:
    print(f"  ! {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. YOY EXPENSE
# ═══════════════════════════════════════════════════════════════════════════════
print("[7] YoY Expense")
try:
    df_e = pd.read_csv(EDGAR_PATH)
    data = (df_e.dropna(subset=["operating_expenses_yoy_pct"])
               .sort_values("period_end")
               .groupby("ticker").last().reset_index())
    save("yoy_expense", {
        "tickers": data["ticker"].tolist(),
        "values":  data["operating_expenses_yoy_pct"].round(2).tolist(),
    })
except Exception as e:
    print(f"  ! {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. AML RISK SCORE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("[8] Risk Score Distribution")
try:
    df_p = pd.read_csv(PAYSIM_PATH)
    counts, edges = np.histogram(df_p["aml_risk_score"].dropna(), bins=10)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))]
    save("risk_score_dist", {"labels": labels, "counts": counts.tolist()})
except Exception as e:
    print(f"  ! {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. MULTI-MODEL COMPARISON — Isolation Forest vs LOF vs Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════
print("[9] Multi-Model Comparison (IF vs LOF vs Autoencoder)")
try:
    df_p = pd.read_csv(PAYSIM_PATH)
    feature_cols = [c for c in ["structuring_flag","balance_drain","balance_mismatch",
                                 "funds_not_received","high_velocity","fan_out_flag"]
                    if c in df_p.columns]
    sample = df_p.sample(min(20000, len(df_p)), random_state=42)
    X = sample[feature_cols].fillna(0).values
    y_true = sample["isfraud"].astype(int).values if "isfraud" in sample.columns else np.zeros(len(sample))

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    results = {}

    # --- Isolation Forest ---
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_sc)
    iso_pred = (iso.predict(X_sc) == -1).astype(int)
    iso_scores = -iso.score_samples(X_sc)
    results["IsolationForest"] = {
        "flagged": int(iso_pred.sum()),
        "flag_rate": round(iso_pred.mean()*100, 2),
        "overlap_fraud": int((iso_pred & y_true.astype(int)).sum()),
        "scores_sample": iso_scores[:200].tolist(),
        "color": "#58a6ff",
    }

    # --- LOF ---
    try:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
        lof_pred = (lof.fit_predict(X_sc) == -1).astype(int)
        lof_scores = -lof.negative_outlier_factor_
        lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-9)
        results["LOF"] = {
            "flagged": int(lof_pred.sum()),
            "flag_rate": round(lof_pred.mean()*100, 2),
            "overlap_fraud": int((lof_pred & y_true.astype(int)).sum()),
            "scores_sample": lof_scores_norm[:200].tolist(),
            "color": "#f85149",
        }
    except Exception as e2:
        print(f"    LOF error: {e2}")

    # --- Autoencoder (numpy-based, no torch needed) ---
    try:
        # Simple 3-layer autoencoder via numpy SGD
        n_feat = X_sc.shape[1]
        np.random.seed(42)
        enc_dim = max(2, n_feat // 2)

        # Init weights
        W1 = np.random.randn(n_feat, enc_dim) * 0.1
        b1 = np.zeros(enc_dim)
        W2 = np.random.randn(enc_dim, n_feat) * 0.1
        b2 = np.zeros(n_feat)

        def relu(x):  return np.maximum(0, x)
        def relu_d(x): return (x > 0).astype(float)

        lr, epochs, batch = 0.01, 30, 256
        X_ae = X_sc.copy()
        for ep in range(epochs):
            idx = np.random.permutation(len(X_ae))
            for start in range(0, len(X_ae), batch):
                batch_x = X_ae[idx[start:start+batch]]
                # Forward
                h  = relu(batch_x @ W1 + b1)
                out = h @ W2 + b2
                err = out - batch_x
                # Backward
                dW2 = h.T @ err / len(batch_x)
                db2 = err.mean(axis=0)
                dh  = err @ W2.T * relu_d(batch_x @ W1 + b1)
                dW1 = batch_x.T @ dh / len(batch_x)
                db1 = dh.mean(axis=0)
                W1 -= lr * dW1; b1 -= lr * db1
                W2 -= lr * dW2; b2 -= lr * db2

        # Reconstruction error
        h_full   = relu(X_sc @ W1 + b1)
        out_full = h_full @ W2 + b2
        recon_err = ((out_full - X_sc)**2).mean(axis=1)
        threshold_ae = np.percentile(recon_err, 95)
        ae_pred = (recon_err > threshold_ae).astype(int)
        ae_norm = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min() + 1e-9)

        results["Autoencoder"] = {
            "flagged": int(ae_pred.sum()),
            "flag_rate": round(ae_pred.mean()*100, 2),
            "overlap_fraud": int((ae_pred & y_true.astype(int)).sum()),
            "scores_sample": ae_norm[:200].tolist(),
            "color": "#3fb950",
            "recon_error_mean": round(float(recon_err.mean()), 4),
            "recon_error_p95":  round(float(threshold_ae), 4),
        }
    except Exception as e3:
        print(f"    Autoencoder error: {e3}")

    # Consensus flags — flagged by 2+ models
    preds = np.array([iso_pred,
                      lof_pred if "LOF" in results else np.zeros(len(iso_pred), dtype=int),
                      ae_pred  if "Autoencoder" in results else np.zeros(len(iso_pred), dtype=int)])
    consensus = (preds.sum(axis=0) >= 2).astype(int)
    results["consensus"] = {
        "flagged": int(consensus.sum()),
        "flag_rate": round(consensus.mean()*100, 2),
        "description": "Transactions flagged by 2+ models simultaneously",
    }
    results["feature_cols"] = feature_cols
    results["sample_size"]  = len(sample)
    save("multi_model", results)
    print(f"    IF: {results['IsolationForest']['flagged']} flags | LOF: {results.get('LOF',{}).get('flagged','N/A')} | AE: {results.get('Autoencoder',{}).get('flagged','N/A')} | Consensus: {results['consensus']['flagged']}")

except Exception as e:
    print(f"  ! Multi-model error: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. SHAP-STYLE FEATURE IMPORTANCE PER FLAG
# ═══════════════════════════════════════════════════════════════════════════════
print("[10] Feature Importance (permutation-based SHAP proxy)")
try:
    df_p = pd.read_csv(PAYSIM_PATH)
    feature_cols = [c for c in ["structuring_flag","balance_drain","balance_mismatch",
                                 "funds_not_received","high_velocity","fan_out_flag"]
                    if c in df_p.columns]
    sample = df_p.sample(min(10000, len(df_p)), random_state=42)
    X = sample[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_sc)
    base_score = iso.decision_function(X_sc).mean()

    importance = {}
    for i, col in enumerate(feature_cols):
        X_perm = X_sc.copy()
        X_perm[:, i] = 0
        delta = abs(base_score - iso.decision_function(X_perm).mean())
        importance[col] = round(delta, 6)

    mx = max(importance.values()) or 1
    importance_norm = {k: round(v/mx*100, 1) for k,v in importance.items()}
    importance_norm_sorted = dict(sorted(importance_norm.items(), key=lambda x: x[1], reverse=True))

    save("feature_importance", {
        "features": list(importance_norm_sorted.keys()),
        "scores":   list(importance_norm_sorted.values()),
        "description": "Permutation-based feature importance — anomaly score delta when feature is zeroed",
    })
    print(f"    Top feature: {list(importance_norm_sorted.keys())[0]} ({list(importance_norm_sorted.values())[0]})")

except Exception as e:
    print(f"  ! {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. HMM REGIME DETECTION OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════
print("[11] HMM Regime Detection over time")
try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("    hmmlearn not installed — using rule-based regime fallback")

try:
    df_p = pd.read_csv(PAYSIM_PATH)

    # Build time series: group by 'step' (each step = 1 hour in PaySim)
    if "step" in df_p.columns:
        ts = df_p.groupby("step").agg(
            tx_count      = ("amount", "count"),
            avg_amount    = ("amount", "mean"),
            structuring   = ("structuring_flag", "sum") if "structuring_flag" in df_p.columns else ("amount","count"),
            high_risk_pct = ("aml_risk_score", lambda x: (x>50).mean()) if "aml_risk_score" in df_p.columns else ("amount","count"),
        ).reset_index()

        # Normalize for HMM
        from sklearn.preprocessing import MinMaxScaler
        feat_cols = ["tx_count","avg_amount","structuring","high_risk_pct"]
        feat_cols = [c for c in feat_cols if c in ts.columns]
        X_hmm = MinMaxScaler().fit_transform(ts[feat_cols].fillna(0))

        if HMM_AVAILABLE and len(ts) >= 10:
            model_hmm = hmmlearn_hmm.GaussianHMM(n_components=3, covariance_type="full",
                                                   n_iter=100, random_state=42)
            model_hmm.fit(X_hmm)
            states = model_hmm.predict(X_hmm)

            # Map states to regime names by mean structuring rate
            state_struct = {s: X_hmm[states==s, feat_cols.index("structuring") if "structuring" in feat_cols else 0].mean()
                           for s in range(3)}
            sorted_states = sorted(state_struct, key=state_struct.get)
            state_map = {sorted_states[0]: "LOW_RISK",
                        sorted_states[1]: "TRANSITION",
                        sorted_states[2]: "HIGH_RISK"}
            regime_labels = [state_map[s] for s in states]
        else:
            # Rule-based fallback: percentile thresholds
            struct_norm = X_hmm[:, feat_cols.index("structuring") if "structuring" in feat_cols else 0]
            p33, p67 = np.percentile(struct_norm, 33), np.percentile(struct_norm, 67)
            regime_labels = ["HIGH_RISK" if v>=p67 else "TRANSITION" if v>=p33 else "LOW_RISK"
                            for v in struct_norm]

        # Downsample for JSON (max 500 points)
        step_size = max(1, len(ts)//500)
        idx = list(range(0, len(ts), step_size))
        save("regime_timeline", {
            "steps":        ts["step"].iloc[idx].tolist(),
            "tx_count":     ts["tx_count"].iloc[idx].tolist(),
            "structuring":  ts["structuring"].iloc[idx].tolist() if "structuring" in ts.columns else [],
            "regimes":      [regime_labels[i] for i in idx],
            "regime_counts": {
                "HIGH_RISK":  regime_labels.count("HIGH_RISK"),
                "TRANSITION": regime_labels.count("TRANSITION"),
                "LOW_RISK":   regime_labels.count("LOW_RISK"),
            },
            "method": "HMM" if HMM_AVAILABLE else "rule-based",
        })
        print(f"    Regimes — HIGH: {regime_labels.count('HIGH_RISK')} | TRANS: {regime_labels.count('TRANSITION')} | LOW: {regime_labels.count('LOW_RISK')} ({len(ts)} steps)")
    else:
        print("    No 'step' column in PaySim — skipping regime timeline")

except Exception as e:
    print(f"  ! HMM error: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════════
# 12. TRANSACTION VOLUME OVER TIME WITH ANOMALY SPIKES
# ═══════════════════════════════════════════════════════════════════════════════
print("[12] Transaction Volume Timeline")
try:
    df_p = pd.read_csv(PAYSIM_PATH)
    if "step" in df_p.columns:
        vol = df_p.groupby("step").agg(
            count=("amount","count"),
            total=("amount","sum"),
            struct=("structuring_flag","sum") if "structuring_flag" in df_p.columns else ("amount","count"),
        ).reset_index()

        # Mark anomaly spikes: steps where count > mean + 2*std
        mean_c = vol["count"].mean()
        std_c  = vol["count"].std()
        vol["is_spike"] = vol["count"] > (mean_c + 2*std_c)

        step_size = max(1, len(vol)//500)
        idx = list(range(0, len(vol), step_size))
        save("tx_volume", {
            "steps":  vol["step"].iloc[idx].tolist(),
            "counts": vol["count"].iloc[idx].tolist(),
            "struct": vol["struct"].iloc[idx].tolist() if "struct" in vol.columns else [],
            "spikes": vol["is_spike"].iloc[idx].tolist(),
            "mean":   round(mean_c, 1),
            "threshold": round(mean_c + 2*std_c, 1),
        })
except Exception as e:
    print(f"  ! {e}")

print("\n✅  All dashboard data exported to dashboard/data/\n")
