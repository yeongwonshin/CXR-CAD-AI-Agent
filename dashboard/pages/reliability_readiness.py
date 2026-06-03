"""
Reliability Readiness — checkpoint-backed dashboard page.

This page is an added Streamlit page only. It does not edit existing dashboard
files. It reads the same checkpoint CSV files already used by the analysis
results dashboard and derives deployment-readiness signals from them.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.reliability.hidden_stratification import detect_hidden_strata
except Exception as exc:  # pragma: no cover - dashboard fallback
    detect_hidden_strata = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

st.set_page_config(
    page_title="CXR-CAD | Reliability Readiness",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 10% 6%, rgba(20,184,166,0.14), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(225,29,72,0.10), transparent 26%),
            linear-gradient(180deg, #f8fbff 0%, #eef7ff 50%, #f8fafc 100%);
    }
    .main .block-container { padding-top: 1.2rem; max-width: 1420px; }
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top left, rgba(56,189,248,0.20), transparent 30%),
            linear-gradient(180deg, #08111f 0%, #0f172a 52%, #111827 100%);
        border-right: 1px solid rgba(148,163,184,0.22);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #e5edf8 !important;
    }
    [data-testid="stSidebarNav"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(148,163,184,0.22) !important; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #f8fafc !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border-color: rgba(148,163,184,0.45) !important;
        opacity: 1 !important;
        font-weight: 650 !important;
    }
    [data-testid="stSidebar"] input:disabled,
    [data-testid="stSidebar"] textarea:disabled {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #f8fafc !important;
        border-color: rgba(148,163,184,0.45) !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #475569 !important;
        fill: #475569 !important;
    }
    div[data-baseweb="popover"] div[role="listbox"] { background: #ffffff !important; }
    div[data-baseweb="popover"] div[role="option"] { color: #0f172a !important; background: #ffffff !important; }
    div[data-baseweb="popover"] div[role="option"]:hover { background: #e0f2fe !important; color: #0f172a !important; }
    .readonly-field {
        background: #f8fafc;
        color: #0f172a !important;
        border: 1px solid rgba(148,163,184,0.45);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-weight: 800;
        font-size: 0.92rem;
        overflow-wrap: anywhere;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
    }
    .readiness-header {
        background:
            radial-gradient(circle at 84% 20%, rgba(125,211,252,0.22), transparent 24%),
            linear-gradient(135deg, #08111f 0%, #15345f 50%, #0f766e 100%);
        border: 1px solid rgba(125,211,252,0.24);
        border-radius: 28px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 24px 58px rgba(15,23,42,0.20);
    }
    .readiness-header .eyebrow { color:#a7f3d0 !important; font-size:0.72rem; letter-spacing:0.16em; text-transform:uppercase; font-weight:800; margin-bottom:0.35rem; }
    .readiness-header h1 { margin:0; font-size:1.72rem; font-weight:850; color:white !important; letter-spacing:-0.03em; }
    .readiness-header p { margin:0.35rem 0 0; font-size:0.9rem; color:#dbeafe !important; line-height:1.5; }
    .quick-guide {
        display:grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.2rem 0 1.05rem;
    }
    .guide-card {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        padding: 1rem 1.05rem;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 16px 36px rgba(15,23,42,0.07);
    }
    .guide-card::after {
        content:"";
        position:absolute;
        inset:auto -22% -44% auto;
        width: 130px;
        height: 130px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(56,189,248,0.22), transparent 65%);
    }
    .guide-card .step {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width: 30px;
        height: 30px;
        border-radius: 999px;
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color:#ffffff;
        font-size:0.78rem;
        font-weight:900;
        box-shadow: 0 8px 20px rgba(37,99,235,0.30);
    }
    .guide-card h4 { margin:0.72rem 0 0.24rem; color:#0f172a !important; font-size:0.96rem; font-weight:900; }
    .guide-card p { margin:0; color:#475569 !important; font-size:0.81rem; line-height:1.48; }
    .readiness-badge {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        padding: 1.18rem 1.25rem;
        margin: 0.7rem 0 0.8rem;
        border: 2px solid rgba(148,163,184,0.24);
        box-shadow: 0 20px 48px rgba(15,23,42,0.16);
    }
    .readiness-badge::before {
        content:"";
        position:absolute;
        inset:0 auto 0 0;
        width: 10px;
        background: #64748b;
    }
    .readiness-badge::after {
        content:"";
        position:absolute;
        right:-58px;
        top:-72px;
        width: 190px;
        height: 190px;
        border-radius: 999px;
        opacity: 0.55;
    }
    .readiness-badge.pass { background: linear-gradient(135deg, #dcfce7, #ecfeff 62%, #eff6ff); border-color: rgba(5,150,105,0.62); }
    .readiness-badge.pass::before { background: linear-gradient(180deg, #059669, #10b981); }
    .readiness-badge.pass::after { background: radial-gradient(circle, rgba(16,185,129,0.28), transparent 67%); }
    .readiness-badge.warning { background: linear-gradient(135deg, #ffedd5, #fef3c7 58%, #fff7ed); border-color: rgba(217,119,6,0.72); }
    .readiness-badge.warning::before { background: linear-gradient(180deg, #d97706, #f59e0b); }
    .readiness-badge.warning::after { background: radial-gradient(circle, rgba(245,158,11,0.34), transparent 67%); }
    .readiness-badge.critical { background: linear-gradient(135deg, #ffe4e6, #fee2e2 55%, #fff1f2); border-color: rgba(190,18,60,0.78); }
    .readiness-badge.critical::before { background: linear-gradient(180deg, #be123c, #ef4444); }
    .readiness-badge.critical::after { background: radial-gradient(circle, rgba(225,29,72,0.35), transparent 67%); }
    .readiness-badge h3 { margin:0; color:#0f172a !important; font-size:1.28rem; font-weight:950; letter-spacing:-0.02em; text-shadow: 0 1px 0 rgba(255,255,255,0.55); }
    .readiness-badge p { margin:0.38rem 0 0; color:#334155 !important; font-size:0.92rem; line-height:1.55; font-weight:750; }
    .readiness-chip {
        display:inline-flex;
        align-items:center;
        gap:0.38rem;
        border-radius: 999px;
        padding:0.26rem 0.62rem;
        margin-bottom:0.52rem;
        background: rgba(15,23,42,0.08);
        color:#0f172a !important;
        font-size:0.74rem;
        font-weight:900;
        letter-spacing:0.08em;
        text-transform: uppercase;
    }
    .judgment-panel {
        border-radius: 22px;
        padding: 1rem 1.12rem;
        margin: -0.15rem 0 1rem;
        border: 1px solid rgba(37,99,235,0.18);
        background:
            radial-gradient(circle at 98% 0%, rgba(56,189,248,0.20), transparent 34%),
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.94));
        box-shadow: 0 16px 38px rgba(15,23,42,0.08);
    }
    .judgment-panel h4 { margin:0 0 0.55rem; color:#0f172a !important; font-size:1rem; font-weight:950; }
    .judgment-list { margin:0; padding-left:1.05rem; color:#334155 !important; font-size:0.86rem; line-height:1.62; }
    .judgment-list li { margin:0.22rem 0; }
    .rr-metric {
        min-height: 126px;
        border-radius: 22px;
        padding: 1rem 1.02rem;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.24);
        box-shadow: 0 16px 34px rgba(15,23,42,0.06);
        position:relative;
        overflow:hidden;
    }
    .rr-metric::after {
        content:"";
        position:absolute;
        right:-46px;
        bottom:-58px;
        width:120px;
        height:120px;
        border-radius:999px;
        background: radial-gradient(circle, rgba(59,130,246,0.15), transparent 70%);
    }
    .rr-metric.good { border-color: rgba(16,185,129,0.32); }
    .rr-metric.warn { border-color: rgba(245,158,11,0.42); }
    .rr-metric.danger { border-color: rgba(225,29,72,0.42); }
    .rr-metric.neutral { border-color: rgba(148,163,184,0.24); }
    .rr-metric .label { color:#64748b !important; font-size:0.74rem; font-weight:850; letter-spacing:0.05em; text-transform:uppercase; }
    .rr-metric .value { color:#0f172a !important; font-size:1.62rem; font-weight:950; letter-spacing:-0.04em; margin-top:0.22rem; }
    .rr-metric .hint { color:#475569 !important; font-size:0.78rem; line-height:1.42; margin-top:0.35rem; font-weight:650; }
    .term-table-wrap {
        border-radius: 20px;
        border: 1px solid rgba(148,163,184,0.20);
        overflow: hidden;
        box-shadow: 0 10px 28px rgba(15,23,42,0.05);
    }
    .term-grid {
        width:100%;
        border-collapse:collapse;
        background:#ffffff;
    }
    .term-grid th {
        background:#0f172a;
        color:#e0f2fe !important;
        font-size:0.78rem;
        text-align:left;
        padding:0.74rem 0.82rem;
    }
    .term-grid td {
        color:#334155 !important;
        font-size:0.80rem;
        line-height:1.48;
        padding:0.74rem 0.82rem;
        border-top:1px solid rgba(226,232,240,0.95);
        vertical-align:top;
    }
    .term-grid td:first-child { font-weight:900; color:#0f172a !important; white-space:nowrap; }
    .issue-help {
        border-radius: 18px;
        padding: 0.82rem 0.95rem;
        background: linear-gradient(135deg, rgba(239,246,255,0.96), rgba(240,253,250,0.92));
        border: 1px solid rgba(14,165,233,0.18);
        color:#334155 !important;
        font-size:0.82rem;
        line-height:1.56;
        margin-top:0.65rem;
    }
    .section-title-note { color:#64748b !important; font-size:0.84rem; line-height:1.55; margin-top:-0.25rem; }
    @media (max-width: 900px) {
        .quick-guide { grid-template-columns: 1fr; }
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 16px 34px rgba(15,23,42,0.06);
    }
    .sidebar-brand {
        border: 1px solid rgba(125,211,252,0.22);
        border-radius: 18px;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,58,138,0.5));
        box-shadow: 0 18px 48px rgba(8,17,31,0.35);
        margin-bottom: 0.8rem;
    }
    .sidebar-brand h2 { margin:0; font-size:1.12rem; color:white !important; font-weight:850; }
    .sidebar-brand p { margin:0.32rem 0 0; color:#b6c7dc !important; font-size:0.78rem; line-height:1.45; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

BASE_CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))

SUPPORTED_MODELS = {
    "densenet":     "DenseNet-121",
    "efficientnet": "EfficientNet-B4",
    "vit":          "ViT-B/16",
}

_selected_model = st.session_state.get("reliability_selected_model", "densenet")
CHECKPOINT_DIR = BASE_CHECKPOINT_DIR / _selected_model

st.markdown(
    f"""
<div class="readiness-header">
    <div class="eyebrow">Reliability readiness</div>
    <h1>CXR-CAD — Reliability Readiness</h1>
    <p>checkpoints/{_selected_model}/ 결과물을 읽어 임상 배치 전 위험 신호를 통합 점검합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)


def _load_csv(name: str) -> pd.DataFrame:
    path = CHECKPOINT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"{name} 로드 실패: {exc}")
        return pd.DataFrame()


def _parse_percent_or_float(value: Any) -> float:
    """Return absolute percentage-point value from '-4.3%', 0.043, or 4.3."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, str):
        txt = value.strip().replace("％", "%")
        if txt in {"", "—", "-", "nan", "NaN"}:
            return float("nan")
        m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
        if not m:
            return float("nan")
        num = float(m.group(0))
        return abs(num) if "%" in txt else abs(num)
    num = float(value)
    return abs(num * 100.0) if abs(num) <= 1.0 else abs(num)


def _binary_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], 0.0, 1.0)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not idx.any():
            continue
        conf = float(p[idx].mean())
        acc = float(y[idx].mean())
        ece += float(idx.mean()) * abs(acc - conf)
    return ece


def _best_youden_j(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], 0.0, 1.0)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    thresholds = np.unique(np.quantile(p, np.linspace(0, 1, 101)))
    best = (-float("inf"), 0.5, float("nan"), float("nan"))
    for thr in thresholds:
        pred = p >= thr
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        j = sens + spec - 1 if np.isfinite(sens) and np.isfinite(spec) else float("nan")
        if np.isfinite(j) and j > best[0]:
            best = (float(j), float(thr), float(sens), float(spec))
    return best


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        y = np.asarray(y_true, dtype=int)
        p = np.asarray(y_prob, dtype=float)
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _shortcut_ratio(region_df: pd.DataFrame) -> float:
    if region_df.empty or "Count" not in region_df.columns:
        return float("nan")
    label_col = region_df.columns[0]
    df = region_df.copy()
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)
    total = float(df["Count"].sum())
    if total <= 0:
        return float("nan")
    normal = df[df[label_col].astype(str).str.contains("lung|정상", case=False, regex=True, na=False)]["Count"].sum()
    return float((total - normal) / total)


def _external_drop_pp(ext_df: pd.DataFrame) -> float:
    if ext_df.empty:
        return float("nan")
    disease_col = "Disease" if "Disease" in ext_df.columns else ext_df.columns[0]
    gap_col = "Gap" if "Gap" in ext_df.columns else None
    if gap_col:
        macro = ext_df[ext_df[disease_col].astype(str).str.lower().isin(["macro_avg", "mean", "average"])]
        if not macro.empty:
            return _parse_percent_or_float(macro.iloc[0][gap_col])
        vals = [_parse_percent_or_float(v) for v in ext_df[gap_col].tolist()]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    if {"NIH AUROC", "CheXpert AUROC"}.issubset(ext_df.columns):
        gap = pd.to_numeric(ext_df["CheXpert AUROC"], errors="coerce") - pd.to_numeric(ext_df["NIH AUROC"], errors="coerce")
        return float(abs(gap.mean()) * 100.0)
    return float("nan")


def _view_gap_pp(view_df: pd.DataFrame) -> float:
    if view_df.empty:
        return float("nan")
    if "Gap vs PA" in view_df.columns:
        vals = [_parse_percent_or_float(v) for v in view_df["Gap vs PA"].tolist()]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            return float(max(vals))
    if "Mean AUROC" in view_df.columns:
        vals = pd.to_numeric(view_df["Mean AUROC"], errors="coerce").dropna().to_numpy()
        if len(vals) >= 2:
            return float((np.max(vals) - np.min(vals)) * 100.0)
    return float("nan")


def _age_gap_pp(age_df: pd.DataFrame) -> float:
    if age_df.empty or "Mean AUROC" not in age_df.columns:
        return float("nan")
    vals = pd.to_numeric(age_df["Mean AUROC"], errors="coerce").dropna().to_numpy()
    if len(vals) < 2:
        return float("nan")
    return float((np.max(vals) - np.min(vals)) * 100.0)


def _gender_gap_pp(gender_df: pd.DataFrame) -> float:
    if gender_df.empty or not {"Male AUROC", "Female AUROC"}.issubset(gender_df.columns):
        return float("nan")
    gap = (pd.to_numeric(gender_df["Male AUROC"], errors="coerce") - pd.to_numeric(gender_df["Female AUROC"], errors="coerce")).abs()
    if gap.dropna().empty:
        return float("nan")
    return float(gap.max() * 100.0)


def _make_proxy_features(pred_df: pd.DataFrame) -> pd.DataFrame:
    feature_parts = []
    if "Patient Age" in pred_df.columns:
        feature_parts.append(pd.to_numeric(pred_df["Patient Age"], errors="coerce").fillna(pred_df["Patient Age"].median()).rename("Patient Age"))
    for col in ["Patient Gender", "View Position"]:
        if col in pred_df.columns:
            feature_parts.append(pd.get_dummies(pred_df[col].fillna("Unknown").astype(str), prefix=col))
    prob_cols = [c for c in pred_df.columns if c.endswith("_prob")]
    if prob_cols:
        feature_parts.append(pred_df[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    if not feature_parts:
        return pd.DataFrame()
    return pd.concat(feature_parts, axis=1)


def _issue(dimension: str, severity: str, message: str, action: str) -> dict[str, str]:
    return {"dimension": dimension, "severity": severity, "message": message, "recommended_action": action}


def _adjustable_report(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    ece = metrics.get("ece")
    if np.isfinite(ece) and ece >= thresholds["ece_warn"]:
        issues.append(_issue("calibration", "warning", f"ECE={ece:.3f} ≥ {thresholds['ece_warn']:.3f}", "Temperature scaling 재보정 및 운영 임계점 재설정"))
    j = metrics.get("youden_j")
    if np.isfinite(j) and j < thresholds["youden_min"]:
        issues.append(_issue("calibration", "warning", f"Youden J={j:.3f} < {thresholds['youden_min']:.3f}", "질환별 threshold 정책 재검토"))
    domain_gap = metrics.get("domain_gap_pp")
    if np.isfinite(domain_gap) and domain_gap >= thresholds["domain_gap_warn_pp"]:
        issues.append(_issue("domain_robustness", "warning", f"Subgroup AUROC gap={domain_gap:.1f}pp", "취약 하위집단 보강, reweighting, subgroup validation 추가"))
    ext_drop = metrics.get("external_drop_pp")
    if np.isfinite(ext_drop) and ext_drop >= thresholds["external_drop_critical_pp"]:
        issues.append(_issue("domain_robustness", "critical", f"External AUROC drop={ext_drop:.1f}pp", "외부기관 fine-tuning/domain adaptation 전 배치 보류"))
    shortcut = metrics.get("shortcut_ratio")
    if np.isfinite(shortcut) and shortcut >= thresholds["shortcut_warn"]:
        issues.append(_issue("localization", "warning", f"Shortcut pattern ratio={shortcut:.1%}", "shortcut-prone sample 정제 및 ROI 기반 검증 추가"))
    hidden_count = metrics.get("hidden_flagged_count")
    if hidden_count is not None and hidden_count > 0:
        issues.append(_issue("hidden_stratification", "warning", f"{hidden_count} underperforming proxy strata detected", "cluster exemplar 확인 및 targeted validation slice 추가"))

    has_critical = any(x["severity"] == "critical" for x in issues)
    has_warning = any(x["severity"] == "warning" for x in issues)
    status = "critical" if has_critical else "warning" if has_warning else "pass"
    return {
        "overall_status": status,
        "deployment_recommendation": {
            "pass": "Proceed with routine monitoring.",
            "warning": "Do not fully deploy until warning items are reviewed.",
            "critical": "Block deployment until critical reliability issues are resolved.",
        }[status],
        "issues": issues,
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _metric_value(value: Any, fmt: str = "{:.3f}", empty: str = "—") -> str:
    if not _finite(value):
        return empty
    return fmt.format(float(value))


def _metric_tone(value: Any, warn_value: float | None = None, direction: str = "low_is_good") -> str:
    if not _finite(value) or warn_value is None:
        return "neutral"
    val = float(value)
    if direction == "low_is_good":
        if val >= warn_value:
            return "warn"
        return "good"
    if val < warn_value:
        return "warn"
    return "good"


def _metric_card(title: str, value: str, hint: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
<div class="rr-metric {tone}">
    <div class="label">{title}</div>
    <div class="value">{value}</div>
    <div class="hint">{hint}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _issue_dimension_ko(dimension: str) -> str:
    return {
        "calibration": "확률 보정성",
        "domain_robustness": "도메인 변화 강건성",
        "localization": "판단 근거 위치 일관성",
        "hidden_stratification": "숨은 취약군(hidden stratification)",
    }.get(dimension, dimension)


def _issue_severity_ko(severity: str) -> str:
    return {"critical": "위험", "warning": "주의", "pass": "양호"}.get(severity, severity)


def _judgment_basis(report: dict[str, Any], metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    issues = report.get("issues", [])
    if not issues:
        return [
            "현재 조절 기준에서 ECE, Youden's J, 외부기관 성능 하락, 하위집단 격차, shortcut 비율이 경고선을 넘지 않았습니다.",
            "따라서 즉시 무제한 운영이 아니라, 정기 모니터링을 붙인 제한적 배치 가능 상태로 해석합니다.",
        ]

    basis = []
    for issue in issues[:5]:
        basis.append(
            f"{_issue_dimension_ko(issue.get('dimension', ''))}: {_issue_severity_ko(issue.get('severity', ''))} — "
            f"{issue.get('message', '')}. 권장 조치: {issue.get('recommended_action', '')}."
        )
    if len(issues) > 5:
        basis.append(f"그 밖에도 {len(issues) - 5}개 항목이 추가로 경고 목록에 포함되었습니다.")
    return basis


def _badge(status: str) -> None:
    if status == "critical":
        title = "CRITICAL — 배치 차단 권고"
        body = "중대한 신뢰성 위험이 기준을 넘었습니다. 문제를 해결하기 전까지 실제 임상 배치를 보류하는 상태입니다."
        chip = "BLOCK DEPLOYMENT"
        css_class = "critical"
    elif status == "warning":
        title = "WARNING — 검토 후 제한 배치 권고"
        body = "주의 항목이 발견되었습니다. 원인 확인·보정·하위집단 재검증 후 제한 배치를 고려해야 합니다."
        chip = "REVIEW BEFORE DEPLOYMENT"
        css_class = "warning"
    else:
        title = "PASS — 정기 모니터링 조건 배치 가능"
        body = "현재 기준에서는 주요 신뢰성 위험이 경고선을 넘지 않았습니다. 단, 배치 후 성능·보정·하위집단 지표는 계속 추적해야 합니다."
        chip = "ROUTINE MONITORING"
        css_class = "pass"
    st.markdown(
        f'<div class="readiness-badge {css_class}"><div class="readiness-chip">{chip}</div><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def _render_judgment_basis(report: dict[str, Any], metrics: dict[str, Any], thresholds: dict[str, Any]) -> None:
    items = "".join(f"<li>{text}</li>" for text in _judgment_basis(report, metrics, thresholds))
    st.markdown(
        f"""
<div class="judgment-panel">
    <h4>판단 근거</h4>
    <ul class="judgment-list">{items}</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_quick_guide() -> None:
    st.markdown(
        """
<div class="quick-guide">
    <div class="guide-card">
        <div class="step">1</div>
        <h4>모델·질환 선택</h4>
        <p>왼쪽에서 DenseNet, EfficientNet, ViT와 판독 대상 질환을 고릅니다. 지표는 선택한 모델·질환 기준으로 즉시 다시 계산됩니다.</p>
    </div>
    <div class="guide-card">
        <div class="step">2</div>
        <h4>경고 기준 조절</h4>
        <p>보수적으로 판단하려면 ECE, subgroup gap, external drop, shortcut 기준을 낮추고, Youden's J 기준은 높입니다.</p>
    </div>
    <div class="guide-card">
        <div class="step">3</div>
        <h4>판단 근거 확인</h4>
        <p>상단 배치 판단 아래의 근거와 Integrated issues를 먼저 보고, 어떤 지표 때문에 PASS/WARNING/CRITICAL인지 확인합니다.</p>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_term_expander() -> None:
    with st.expander("용어설명 자세히 보기", expanded=False):
        st.markdown(
            """
<div class="term-table-wrap">
<table class="term-grid">
    <thead><tr><th>용어</th><th>쉬운 의미</th><th>사용자가 조절할 때의 해석</th></tr></thead>
    <tbody>
        <tr><td>AUROC</td><td>양성/음성 환자를 얼마나 잘 구분하는지 보는 전체 판별력입니다. 1에 가까울수록 좋습니다.</td><td>직접 조절하는 값은 아니며, 모델 성능을 읽는 핵심 결과값입니다.</td></tr>
        <tr><td>ECE</td><td>모델이 “80% 확률”이라고 말했을 때 실제로도 비슷한 비율로 맞는지 보는 확률 보정 오차입니다. 낮을수록 좋습니다.</td><td>ECE warning threshold를 낮추면 더 엄격해지고, 높이면 더 관대해집니다. 의료 배치 전에는 보통 낮게 잡는 편이 안전합니다.</td></tr>
        <tr><td>Youden's J</td><td>민감도와 특이도를 함께 고려한 임계값 품질 지표입니다. 높을수록 양성과 음성을 균형 있게 나눈다는 뜻입니다.</td><td>Minimum Youden's J를 높이면 “충분히 좋은 운영 임계값”만 통과시킵니다.</td></tr>
        <tr><td>Threshold</td><td>질환 확률이 몇 % 이상이면 양성으로 볼지 정하는 기준선입니다.</td><td>놓치면 안 되는 질환은 낮게, 과잉 알림을 줄이고 싶으면 높게 운영합니다.</td></tr>
        <tr><td>Sensitivity</td><td>실제 질환자를 양성으로 잡아내는 비율입니다. 높을수록 놓침이 줄어듭니다.</td><td>응급·중증 질환에서는 민감도를 더 중시합니다.</td></tr>
        <tr><td>Specificity</td><td>실제 정상/음성인 경우를 음성으로 잘 걸러내는 비율입니다. 높을수록 불필요한 알림이 줄어듭니다.</td><td>검수 부담이 큰 운영 환경에서는 특이도도 함께 봐야 합니다.</td></tr>
        <tr><td>Subgroup gap</td><td>나이, 성별, 촬영 자세 같은 하위집단 사이의 AUROC 차이입니다. 작을수록 특정 집단에 덜 치우칩니다.</td><td>Subgroup gap warning threshold를 낮추면 하위집단 불균형에 더 민감하게 경고합니다.</td></tr>
        <tr><td>External drop</td><td>다른 병원/외부 데이터에서 성능이 얼마나 떨어지는지 보는 값입니다.</td><td>External drop critical threshold를 낮추면 외부기관 일반화 실패를 더 강하게 차단합니다.</td></tr>
        <tr><td>Shortcut ratio</td><td>폐 병변 대신 마커, 테두리, 문자, 기기 흔적 같은 엉뚱한 단서에 의존할 위험 신호입니다.</td><td>Shortcut ratio warning threshold를 낮추면 Grad-CAM/ROI 검증을 더 엄격하게 요구합니다.</td></tr>
        <tr><td>Hidden stratification</td><td>전체 평균은 좋아도 특정 숨은 환자군에서만 성능이 나쁜지 찾는 검사입니다.</td><td>Cluster count를 늘리면 더 세분화해서 찾고, Minimum stratum size를 높이면 너무 작은 군집을 무시합니다.</td></tr>
        <tr><td>pp</td><td>percentage point의 약자입니다. 예: 90%에서 87%로 떨어지면 3pp 하락입니다.</td><td>AUROC gap/drop 같은 차이를 읽을 때 쓰는 단위입니다.</td></tr>
    </tbody>
</table>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_control_help(thresholds: dict[str, Any]) -> None:
    st.markdown(
        f"""
<div class="issue-help">
    <b>조절 방법:</b> 현재 기준은 ECE ≥ {thresholds['ece_warn']:.3f}, Youden's J &lt; {thresholds['youden_min']:.2f},
    하위집단 격차 ≥ {thresholds['domain_gap_warn_pp']:.1f}pp, 외부기관 하락 ≥ {thresholds['external_drop_critical_pp']:.1f}pp,
    shortcut 비율 ≥ {thresholds['shortcut_warn']:.0%}일 때 경고를 띄웁니다.
</div>
""",
        unsafe_allow_html=True,
    )


op_df = _load_csv("op_analysis.csv")
gender_df = _load_csv("gender_subgroup.csv")
age_df = _load_csv("age_subgroup.csv")
view_df = _load_csv("view_subgroup.csv")
ext_df = _load_csv("domain_shift.csv")
region_df = _load_csv("shortcut_regions.csv")
pred_df = _load_csv("test_predictions.csv")

true_cols = sorted([c[:-5] for c in pred_df.columns if c.endswith("_true")]) if not pred_df.empty else []
prob_cols = sorted([c[:-5] for c in pred_df.columns if c.endswith("_prob")]) if not pred_df.empty else []
diseases = [d for d in true_cols if d in prob_cols]
default_disease = "Cardiomegaly" if "Cardiomegaly" in diseases else diseases[0] if diseases else None

with st.sidebar:
    st.markdown("""<div class="sidebar-brand"><h2>Reliability Readiness</h2><p>배치 전 신뢰성 위험 신호를 점검합니다.</p></div>""", unsafe_allow_html=True)
    st.markdown("### 분석 모델")
    st.selectbox(
        "결과를 볼 모델 선택",
        options=list(SUPPORTED_MODELS.keys()),
        format_func=lambda k: SUPPORTED_MODELS[k],
        key="reliability_selected_model",
    )
    st.divider()
    st.markdown("### Data source")
    st.markdown(f'<div class="readonly-field">{CHECKPOINT_DIR}</div>', unsafe_allow_html=True)
    loaded = [name for name, df in [
        ("op_analysis.csv", op_df),
        ("test_predictions.csv", pred_df),
        ("view_subgroup.csv", view_df),
        ("age_subgroup.csv", age_df),
        ("gender_subgroup.csv", gender_df),
        ("domain_shift.csv", ext_df),
        ("shortcut_regions.csv", region_df),
    ] if not df.empty]
    st.caption("Loaded: " + (", ".join(loaded) if loaded else "none"))

    if diseases:
        disease = st.selectbox("Target disease", diseases, index=diseases.index(default_disease))
    else:
        disease = None
        st.warning("test_predictions.csv에서 *_true / *_prob 컬럼을 찾지 못했습니다.")

    st.markdown("### 배치 판단 기준")
    st.caption("기준을 낮추면 더 엄격하게, 높이면 더 관대하게 판정합니다. 단, Youden's J는 높일수록 더 엄격합니다.")
    ece_warn = st.slider("ECE 경고 기준", 0.00, 1.00, 0.05, 0.01, help="확률 보정 오차입니다. 낮을수록 좋고, 이 값 이상이면 경고로 봅니다.")
    youden_min = st.slider("최소 Youden's J", 0.00, 1.00, 0.60, 0.01, help="민감도와 특이도를 함께 보는 운영 임계값 품질입니다. 이 값보다 낮으면 경고로 봅니다.")
    domain_gap_warn_pp = st.slider("하위집단 격차 경고 기준 (pp)", 0.0, 15.0, 3.0, 0.5, help="나이/성별/촬영자세 집단 사이 AUROC 차이입니다. 이 값 이상이면 경고로 봅니다.")
    external_drop_critical_pp = st.slider("외부기관 성능 하락 차단 기준 (pp)", 0.0, 15.0, 3.0, 0.5, help="외부 데이터에서 AUROC가 이 값 이상 떨어지면 CRITICAL로 봅니다.")
    shortcut_warn = st.slider("Shortcut 비율 경고 기준", 0.00, 0.50, 0.05, 0.01, help="폐 영역 밖 단서에 의존할 위험 비율입니다. 이 값 이상이면 경고로 봅니다.")

    st.markdown("### 숨은 취약군 탐색")
    n_clusters = st.slider("군집 개수", 2, 8, 4, 1, help="더 크게 잡으면 환자군을 더 세분화해 숨은 취약군을 찾습니다.")
    min_size = st.slider("최소 군집 크기", 10, 200, 20, 10, help="너무 작은 군집을 경고에서 제외하기 위한 최소 표본 수입니다.")

metrics: dict[str, Any] = {}
threshold_for_error = 0.5
hidden_result = None
hidden_note = ""

if disease and not pred_df.empty:
    y_true = pd.to_numeric(pred_df[f"{disease}_true"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob = pd.to_numeric(pred_df[f"{disease}_prob"], errors="coerce").fillna(0.0).clip(0, 1).to_numpy()
    ece = _binary_ece(y_true, y_prob)
    j, best_thr, sens, spec = _best_youden_j(y_true, y_prob)
    auc = _safe_auroc(y_true, y_prob)
    metrics.update({"ece": ece, "youden_j": j, "best_threshold": best_thr, "sensitivity": sens, "specificity": spec, "auroc": auc})
    threshold_for_error = best_thr if np.isfinite(best_thr) else 0.5

    features = _make_proxy_features(pred_df)
    if detect_hidden_strata is not None and not features.empty and len(features) >= n_clusters:
        try:
            hidden_result = detect_hidden_strata(
                features.to_numpy(dtype=np.float32),
                y_true,
                y_prob,
                n_clusters=n_clusters,
                threshold=threshold_for_error,
                min_size=min_size,
                random_state=42,
            )
            metrics["hidden_flagged_count"] = hidden_result["flagged_count"]
            hidden_note = "현재 프로젝트에는 penultimate embedding 파일이 없어 metadata+probability 기반 proxy clustering으로 계산했습니다."
        except Exception as exc:
            hidden_note = f"hidden stratification 계산 실패: {exc}"
    elif IMPORT_ERROR is not None:
        hidden_note = f"hidden stratification 모듈 import 실패: {IMPORT_ERROR}"

view_gap = _view_gap_pp(view_df)
age_gap = _age_gap_pp(age_df)
gender_gap = _gender_gap_pp(gender_df)
finite_gaps = [x for x in [view_gap, age_gap, gender_gap] if np.isfinite(x)]
metrics["domain_gap_pp"] = max(finite_gaps) if finite_gaps else float("nan")
metrics["external_drop_pp"] = _external_drop_pp(ext_df)
metrics["shortcut_ratio"] = _shortcut_ratio(region_df)

thresholds = {
    "ece_warn": ece_warn,
    "youden_min": youden_min,
    "domain_gap_warn_pp": domain_gap_warn_pp,
    "external_drop_critical_pp": external_drop_critical_pp,
    "shortcut_warn": shortcut_warn,
}
report = _adjustable_report(metrics, thresholds)

_render_quick_guide()
_render_term_expander()
_render_control_help(thresholds)

_badge(report["overall_status"])
_render_judgment_basis(report, metrics, thresholds)

st.subheader("핵심 지표 한눈에 보기")
st.markdown(
    '<p class="section-title-note">수치와 하단의 해석을 함께 확인하세요.</p>',
    unsafe_allow_html=True,
)

row1 = st.columns(4)
with row1[0]:
    _metric_card(
        "Target AUROC",
        _metric_value(metrics.get("auroc"), "{:.3f}"),
        "선택 질환의 전체 판별력입니다. 1에 가까울수록 좋습니다.",
        _metric_tone(metrics.get("auroc"), 0.80, "high_is_good"),
    )
with row1[1]:
    _metric_card(
        "ECE",
        _metric_value(metrics.get("ece"), "{:.3f}"),
        f"확률 보정 오차입니다. 현재 경고 기준은 {ece_warn:.3f}입니다.",
        _metric_tone(metrics.get("ece"), ece_warn, "low_is_good"),
    )
with row1[2]:
    _metric_card(
        "Best Youden's J",
        _metric_value(metrics.get("youden_j"), "{:.3f}"),
        f"민감도·특이도 균형입니다. 최소 기준은 {youden_min:.2f}입니다.",
        _metric_tone(metrics.get("youden_j"), youden_min, "high_is_good"),
    )
with row1[3]:
    _metric_card(
        "Shortcut ratio",
        _metric_value(metrics.get("shortcut_ratio"), "{:.1%}"),
        f"폐 밖 단서 의존 위험입니다. 경고 기준은 {shortcut_warn:.0%}입니다.",
        _metric_tone(metrics.get("shortcut_ratio"), shortcut_warn, "low_is_good"),
    )

row2 = st.columns(4)
with row2[0]:
    _metric_card(
        "View gap",
        _metric_value(view_gap, "{:.1f}pp"),
        f"촬영 자세별 성능 차이입니다. 기준은 {domain_gap_warn_pp:.1f}pp입니다.",
        _metric_tone(view_gap, domain_gap_warn_pp, "low_is_good"),
    )
with row2[1]:
    _metric_card(
        "Age gap",
        _metric_value(age_gap, "{:.1f}pp"),
        f"연령대별 성능 차이입니다. 기준은 {domain_gap_warn_pp:.1f}pp입니다.",
        _metric_tone(age_gap, domain_gap_warn_pp, "low_is_good"),
    )
with row2[2]:
    _metric_card(
        "External drop",
        _metric_value(metrics.get("external_drop_pp"), "{:.1f}pp"),
        f"외부기관 성능 하락입니다. 차단 기준은 {external_drop_critical_pp:.1f}pp입니다.",
        _metric_tone(metrics.get("external_drop_pp"), external_drop_critical_pp, "low_is_good"),
    )
with row2[3]:
    hidden_count = metrics.get("hidden_flagged_count", "—")
    hidden_tone = "warn" if isinstance(hidden_count, (int, float)) and hidden_count > 0 else "good" if hidden_count == 0 else "neutral"
    _metric_card(
        "Hidden strata",
        str(hidden_count),
        "평균 성능에 가려진 취약 환자군 개수입니다.",
        hidden_tone,
    )

st.divider()

left, right = st.columns([1, 1])
with left:
    st.subheader("계산에 사용된 입력 파일")
    st.markdown(
        '<p class="section-title-note">각 지표가 어떤 CSV에서 계산되었는지 보여줍니다. 값이 비어 있으면 해당 파일 또는 컬럼이 없다는 뜻입니다.</p>',
        unsafe_allow_html=True,
    )
    source_rows = [
        {"지표": "확률 보정 오차(ECE)", "입력 파일": "test_predictions.csv", "계산값": None if not np.isfinite(metrics.get("ece", float("nan"))) else round(metrics["ece"], 4)},
        {"지표": "운영 임계값 품질(Youden J)", "입력 파일": "test_predictions.csv", "계산값": None if not np.isfinite(metrics.get("youden_j", float("nan"))) else f"J={metrics['youden_j']:.3f}, threshold={metrics['best_threshold']:.3f}"},
        {"지표": "하위집단 AUROC 격차", "입력 파일": "view/age/gender_subgroup.csv", "계산값": None if not np.isfinite(metrics.get("domain_gap_pp", float("nan"))) else f"{metrics['domain_gap_pp']:.1f}pp"},
        {"지표": "외부기관 AUROC 하락", "입력 파일": "domain_shift.csv", "계산값": None if not np.isfinite(metrics.get("external_drop_pp", float("nan"))) else f"{metrics['external_drop_pp']:.1f}pp"},
        {"지표": "Shortcut 위험 비율", "입력 파일": "shortcut_regions.csv", "계산값": None if not np.isfinite(metrics.get("shortcut_ratio", float("nan"))) else f"{metrics['shortcut_ratio']:.1%}"},
        {"지표": "숨은 취약군 탐색", "입력 파일": "test_predictions.csv proxy features", "계산값": metrics.get("hidden_flagged_count", None)},
    ]
    df_source = pd.DataFrame(source_rows).astype(str)
    st.dataframe(df_source, hide_index=True, width="stretch")

with right:
    st.subheader("우선 확인할 문제 항목")
    st.markdown(
        '<p class="section-title-note">여기에 항목이 있으면 상단 배치 판단이 WARNING 또는 CRITICAL로 바뀝니다. 권장 조치부터 처리하면 됩니다.</p>',
        unsafe_allow_html=True,
    )
    issues_df = pd.DataFrame(report["issues"])
    if issues_df.empty:
        st.success("현재 조절 기준에서는 경고 항목이 없습니다.")
    else:
        issues_view = issues_df.assign(
            구분=issues_df["dimension"].map(_issue_dimension_ko),
            심각도=issues_df["severity"].map(_issue_severity_ko),
        )[["심각도", "구분", "message", "recommended_action"]].rename(
            columns={"message": "판단 근거", "recommended_action": "권장 조치"}
        )
        st.dataframe(issues_view, hide_index=True, width="stretch")

st.divider()

if disease and not pred_df.empty:
    st.subheader(f"확률 보정과 운영 기준선 — {disease}")
    st.markdown(
        '<p class="section-title-note">점이 대각선에 가까울수록 모델의 예측 확률과 실제 양성 비율이 잘 맞습니다. 대각선에서 크게 벗어나면 ECE가 커지고, 배치 전 보정이 필요합니다.</p>',
        unsafe_allow_html=True,
    )
    y_true = pd.to_numeric(pred_df[f"{disease}_true"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob = pd.to_numeric(pred_df[f"{disease}_prob"], errors="coerce").fillna(0.0).clip(0, 1).to_numpy()
    bins = np.linspace(0, 1, 11)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if idx.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(idx.sum()), "mean_prob": float(y_prob[idx].mean()), "observed_rate": float(y_true[idx].mean())})
    cal_df = pd.DataFrame(rows)
    if not cal_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal_df["mean_prob"], y=cal_df["observed_rate"], mode="markers+lines", name="Observed"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="평균 예측 확률", yaxis_title="실제 양성 비율", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.divider()
st.subheader("숨은 취약군(hidden stratification) 탐색")
st.markdown(
    '<p class="section-title-note">전체 평균 성능은 좋아도 특정 조건의 환자군에서만 오류가 집중될 수 있습니다.</p>',
    unsafe_allow_html=True,
)
if hidden_note:
    st.info(hidden_note)
if hidden_result is not None:
    strata_df = pd.DataFrame([x.__dict__ for x in hidden_result["strata"]])
    if not strata_df.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(strata_df, hide_index=True, width="stretch")
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=strata_df["stratum_id"].astype(str), y=strata_df["error_rate"], name="Error rate"))
            fig.add_trace(go.Scatter(x=strata_df["stratum_id"].astype(str), y=strata_df["auroc"], mode="lines+markers", name="AUROC", yaxis="y2"))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Proxy stratum", yaxis=dict(title="오류율", range=[0, 1]), yaxis2=dict(title="AUROC", overlaying="y", side="right", range=[0, 1]), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Minimum size 조건을 만족하는 strata가 없습니다.")
else:
    st.warning("hidden stratification proxy 결과를 계산하지 못했습니다.")

st.divider()
with st.expander("개발자용 원본 readiness JSON 보기", expanded=False):
    st.code(json.dumps({"metrics": metrics, "thresholds": thresholds, "report": report}, indent=2, ensure_ascii=False, default=str), language="json")

st.caption(
    "주의: 실제 ROI heatmap/mask 파일이 없는 현재 프로젝트에서는 ROI energy를 새로 계산하지 않고, "
    "shortcut_regions.csv의 폐 영역 이탈 집계를 localization 위험 신호로 사용합니다. "
    "추후 Grad-CAM heatmap과 lung/lesion ROI mask를 저장하면 ROI consistency를 완전 자동화할 수 있습니다."
)
