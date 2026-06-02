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
    .readiness-badge {
        border-radius: 18px;
        padding: 1rem 1.15rem;
        margin: 0.7rem 0 0.8rem;
        border: 1px solid rgba(148,163,184,0.24);
        box-shadow: 0 16px 34px rgba(15,23,42,0.08);
    }
    .readiness-badge.pass { background: linear-gradient(135deg, #ecfdf5, #dbeafe); border-color: rgba(20,184,166,0.32); }
    .readiness-badge.warning { background: linear-gradient(135deg, #fff7ed, #fef3c7); border-color: rgba(245,158,11,0.44); }
    .readiness-badge.critical { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border-color: rgba(225,29,72,0.36); }
    .readiness-badge h3 { margin:0; color:#0f172a !important; font-size:1rem; font-weight:850; }
    .readiness-badge p { margin:0.32rem 0 0; color:#475569 !important; font-size:0.86rem; line-height:1.48; }
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


def _badge(status: str) -> None:
    if status == "critical":
        title = "CRITICAL — 배치 차단 권고"
        body = "중대한 신뢰성 위험이 있어 문제를 해결하기 전까지 배치를 보류하는 상태입니다."
        css_class = "critical"
    elif status == "warning":
        title = "WARNING — 검토 후 제한 배치 권고"
        body = "경고 항목 검토와 보정 후 제한적 배치를 고려해야 하는 상태입니다."
        css_class = "warning"
    else:
        title = "PASS — routine monitoring 권고"
        body = "현재 기준에서는 정기 모니터링 조건으로 배치 가능성이 높은 상태입니다."
        css_class = "pass"
    st.markdown(f'<div class="readiness-badge {css_class}"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


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

    st.markdown("### Risk thresholds")
    ece_warn = st.slider("ECE warning threshold", 0.00, 0.20, 0.05, 0.005)
    youden_min = st.slider("Minimum Youden's J", 0.00, 1.00, 0.60, 0.01)
    domain_gap_warn_pp = st.slider("Subgroup gap warning threshold (pp)", 0.0, 15.0, 3.0, 0.5)
    external_drop_critical_pp = st.slider("External drop critical threshold (pp)", 0.0, 15.0, 3.0, 0.5)
    shortcut_warn = st.slider("Shortcut ratio warning threshold", 0.00, 0.50, 0.05, 0.01)

    st.markdown("### Hidden-strata proxy")
    n_clusters = st.slider("Cluster count", 2, 8, 4, 1)
    min_size = st.slider("Minimum stratum size", 10, 200, 20, 10)

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

_badge(report["overall_status"])
st.write(report["deployment_recommendation"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Target AUROC", "—" if not np.isfinite(metrics.get("auroc", float("nan"))) else f"{metrics['auroc']:.3f}")
m2.metric("ECE", "—" if not np.isfinite(metrics.get("ece", float("nan"))) else f"{metrics['ece']:.3f}")
m3.metric("Best Youden's J", "—" if not np.isfinite(metrics.get("youden_j", float("nan"))) else f"{metrics['youden_j']:.3f}")
m4.metric("Shortcut ratio", "—" if not np.isfinite(metrics.get("shortcut_ratio", float("nan"))) else f"{metrics['shortcut_ratio']:.1%}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("View gap", "—" if not np.isfinite(view_gap) else f"{view_gap:.1f}pp")
m6.metric("Age gap", "—" if not np.isfinite(age_gap) else f"{age_gap:.1f}pp")
m7.metric("External drop", "—" if not np.isfinite(metrics.get("external_drop_pp", float("nan"))) else f"{metrics['external_drop_pp']:.1f}pp")
m8.metric("Hidden strata flagged", metrics.get("hidden_flagged_count", "—"))

st.divider()

left, right = st.columns([1, 1])
with left:
    st.subheader("Checkpoint-backed inputs")
    source_rows = [
        {"signal": "Calibration ECE", "source": "test_predictions.csv", "value": None if not np.isfinite(metrics.get("ece", float("nan"))) else round(metrics["ece"], 4)},
        {"signal": "Youden J / threshold", "source": "test_predictions.csv", "value": None if not np.isfinite(metrics.get("youden_j", float("nan"))) else f"J={metrics['youden_j']:.3f}, threshold={metrics['best_threshold']:.3f}"},
        {"signal": "Subgroup AUROC gap", "source": "view/age/gender_subgroup.csv", "value": None if not np.isfinite(metrics.get("domain_gap_pp", float("nan"))) else f"{metrics['domain_gap_pp']:.1f}pp"},
        {"signal": "External AUROC drop", "source": "domain_shift.csv", "value": None if not np.isfinite(metrics.get("external_drop_pp", float("nan"))) else f"{metrics['external_drop_pp']:.1f}pp"},
        {"signal": "Shortcut pattern ratio", "source": "shortcut_regions.csv", "value": None if not np.isfinite(metrics.get("shortcut_ratio", float("nan"))) else f"{metrics['shortcut_ratio']:.1%}"},
        {"signal": "Hidden stratification", "source": "test_predictions.csv proxy features", "value": metrics.get("hidden_flagged_count", None)},
    ]
    df_source = pd.DataFrame(source_rows).astype(str)
    st.dataframe(df_source, hide_index=True, width="stretch")

with right:
    st.subheader("Integrated readiness issues")
    issues_df = pd.DataFrame(report["issues"])
    if issues_df.empty:
        st.success("No reliability issues under the current thresholds.")
    else:
        st.dataframe(issues_df, hide_index=True, width="stretch")

st.divider()

if disease and not pred_df.empty:
    st.subheader(f"Calibration and operating point — {disease}")
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
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Mean predicted probability", yaxis_title="Observed positive rate", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.divider()
st.subheader("Hidden stratification proxy")
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
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Proxy stratum", yaxis=dict(title="Error rate", range=[0, 1]), yaxis2=dict(title="AUROC", overlaying="y", side="right", range=[0, 1]), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Minimum size 조건을 만족하는 strata가 없습니다.")
else:
    st.warning("hidden stratification proxy 결과를 계산하지 못했습니다.")

st.divider()
st.subheader("Raw readiness JSON")
st.code(json.dumps({"metrics": metrics, "thresholds": thresholds, "report": report}, indent=2, ensure_ascii=False, default=str), language="json")

st.caption(
    "주의: 실제 ROI heatmap/mask 파일이 없는 현재 프로젝트에서는 ROI energy를 새로 계산하지 않고, "
    "shortcut_regions.csv의 폐 영역 이탈 집계를 localization 위험 신호로 사용합니다. "
    "추후 Grad-CAM heatmap과 lung/lesion ROI mask를 저장하면 ROI consistency를 완전 자동화할 수 있습니다."
)
