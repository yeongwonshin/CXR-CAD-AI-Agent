"""🧠 Agentic Case Workbench — MedRAX-style multi-image CXR workflow.

기존 CXR-CAD 모델 학습 결과를 그대로 사용하면서, 플랫폼 레벨에서
MedRAX의 강점인 multi-image 입력, tool trace, DICOM 처리, 케이스 비교,
이미지별 판독문 초안/의료진 피드백을 제공하는 Streamlit 페이지입니다.
"""

from __future__ import annotations

import base64
import io
import os
from html import escape
from typing import Any, Dict, Iterable, List

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

DISEASE_LABELS_KR = {
    "Atelectasis": "무기폐",
    "Cardiomegaly": "심비대",
    "Effusion": "흉수",
    "Infiltration": "폐 침윤",
    "Mass": "종괴",
    "Nodule": "결절",
    "Pneumonia": "폐렴",
    "Pneumothorax": "기흉",
    "Consolidation": "경화",
    "Edema": "폐부종",
    "Emphysema": "폐기종",
    "Fibrosis": "섬유화",
    "Pleural_Thickening": "흉막 비후",
    "Hernia": "탈장",
}

MODEL_OPTIONS = {
    "ensemble": "Ensemble (Recommended)",
    "densenet": "DenseNet-121",
    "efficientnet": "EfficientNet-B4",
    "vit": "ViT-B/16",
}

FEEDBACK_TYPES = [
    "AI 판단 동의",
    "AI 판단 불일치",
    "히트맵 위치 부정확",
    "질환 라벨 수정",
    "판독의 코멘트",
]

st.set_page_config(
    page_title="CXR-CAD | Agentic Case Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 8%, rgba(37,99,235,0.15), transparent 28%),
        radial-gradient(circle at 92% 14%, rgba(20,184,166,0.14), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, #eef7ff 52%, #f8fafc 100%);
}
.main .block-container { max-width: 1460px; padding-top: 1.2rem; }
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, rgba(56,189,248,0.22), transparent 30%), linear-gradient(180deg, #08111f 0%, #0f172a 52%, #111827 100%);
}
[data-testid="stSidebar"] * { color: #e5edf8 !important; }
.agent-header {
    position: relative; overflow:hidden; border-radius: 30px; padding: 1.65rem 2rem; margin-bottom: 1.1rem;
    background: radial-gradient(circle at 83% 18%, rgba(125,211,252,0.26), transparent 25%), linear-gradient(135deg, #08111f 0%, #12345f 50%, #0f766e 100%);
    border: 1px solid rgba(125,211,252,0.26); box-shadow: 0 24px 58px rgba(15,23,42,0.20); color:white;
}
.agent-header .eyebrow { color:#a7f3d0 !important; font-size:0.72rem; letter-spacing:0.16em; text-transform:uppercase; font-weight:900; margin-bottom:0.35rem; }
.agent-header h1 { margin:0; font-size:1.85rem; font-weight:900; color:white !important; letter-spacing:-0.04em; }
.agent-header p { margin:0.45rem 0 0; font-size:0.92rem; color:#dbeafe !important; line-height:1.55; max-width: 980px; }
.agent-pill { display:inline-flex; align-items:center; gap:0.35rem; padding:0.34rem 0.72rem; border-radius:999px; margin:0.85rem 0.35rem 0 0; background:rgba(255,255,255,0.11); border:1px solid rgba(255,255,255,0.18); color:#e0f2fe !important; font-size:0.75rem; font-weight:800; }
.agent-card {
    background: rgba(255,255,255,0.94); border: 1px solid rgba(148,163,184,0.24); border-radius: 22px; padding: 1.1rem 1.2rem;
    box-shadow: 0 18px 42px rgba(15,23,42,0.08); margin-bottom: 1rem; backdrop-filter: blur(10px);
}
.agent-card h3, .agent-card h4 { margin:0 0 0.5rem; color:#0f172a !important; font-weight:900; }
.agent-card p { color:#475569 !important; font-size:0.88rem; line-height:1.55; }
.section-title { font-size:1.05rem; font-weight:900; color:#0f172a; margin:1.05rem 0 0.55rem; display:flex; align-items:center; gap:0.5rem; }
.section-title::before { content:""; width:0.56rem; height:0.56rem; border-radius:999px; background:linear-gradient(135deg,#2563eb,#14b8a6); box-shadow:0 0 0 5px rgba(14,165,233,0.12); }
.metric-card { border-radius:20px; padding:1rem; background:linear-gradient(135deg,#ffffff,#eff6ff); border:1px solid rgba(125,211,252,0.35); box-shadow:0 14px 32px rgba(15,23,42,0.08); }
.metric-card .value { font-size:1.75rem; font-weight:900; color:#0f172a !important; letter-spacing:-0.04em; }
.metric-card .label { font-size:0.72rem; color:#64748b !important; font-weight:800; letter-spacing:0.07em; text-transform:uppercase; }
.triage { border-radius:18px; padding:1rem 1.1rem; border:1px solid rgba(148,163,184,0.22); margin-bottom:0.9rem; }
.triage h4 { margin:0; font-size:1.05rem; font-weight:900; color:#0f172a !important; }
.triage p { margin:0.35rem 0 0; color:#334155 !important; line-height:1.5; }
.triage.urgent { background:linear-gradient(135deg,#ffe4e6,#fee2e2); border-color:rgba(225,29,72,0.45); }
.triage.high { background:linear-gradient(135deg,#ffedd5,#fef3c7); border-color:rgba(245,158,11,0.45); }
.triage.normal { background:linear-gradient(135deg,#ecfeff,#eff6ff); border-color:rgba(14,165,233,0.35); }
.triage.demo { background:linear-gradient(135deg,#eef2ff,#f5f3ff); border-color:rgba(124,58,237,0.35); }
.disease-tag { display:inline-flex; margin:0.18rem; padding:0.32rem 0.62rem; border-radius:999px; background:linear-gradient(135deg,#dbeafe,#ccfbf1); color:#0f172a !important; font-size:0.77rem; font-weight:850; border:1px solid rgba(14,165,233,0.22); }
.trace-step { border-left:4px solid #0ea5e9; padding:0.55rem 0.75rem; margin:0.35rem 0; background:rgba(239,246,255,0.9); border-radius:0 14px 14px 0; }
.trace-step b { color:#0f172a !important; }
.small-muted { color:#64748b !important; font-size:0.78rem; line-height:1.45; }
</style>
""",
    unsafe_allow_html=True,
)


def _guess_content_type(filename: str, fallback: str | None) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith((".dcm", ".dicom")):
        return "application/dicom"
    return fallback or "application/octet-stream"


def call_agent_api(files, model_key: str, threshold: float, question: str) -> dict | None:
    multipart = []
    for f in files:
        raw = f.getvalue()
        multipart.append(("files", (f.name, raw, _guess_content_type(f.name, getattr(f, "type", None)))))
    try:
        resp = requests.post(
            f"{API_URL}/agent/analyze",
            params={"model": model_key, "threshold": threshold, "question": question},
            files=multipart,
            timeout=240,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Agent API 오류 {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("백엔드 API에 연결할 수 없습니다. `uvicorn api.main:app --reload --port 8000`을 먼저 실행하세요.")
    except Exception as exc:
        st.error(f"Agent 분석 요청 중 오류가 발생했습니다: {exc}")
    return None


def call_feedback_api(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"피드백 저장 오류 {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("백엔드 API에 연결할 수 없어 피드백을 저장하지 못했습니다.")
    return None


def check_api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def render_bar_chart(probs: Dict[str, float], threshold: float) -> go.Figure:
    items = sorted(probs.items(), key=lambda item: item[1])
    labels = [DISEASE_LABELS_KR.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>확률: %{x:.2%}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line=dict(width=2, dash="dash"), annotation_text=f"임계값 {threshold:.0%}")
    fig.update_layout(
        height=430,
        margin=dict(l=0, r=50, t=20, b=10),
        xaxis=dict(range=[0, 1.1], tickformat=".0%", gridcolor="rgba(148,163,184,0.18)"),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter"),
    )
    return fig


def prediction_summary_payload(case: dict, edited_report: str) -> dict:
    prediction = case.get("prediction", {}) or {}
    return {
        "filename": case.get("filename"),
        "case_id": case.get("case_id"),
        "top_disease": case.get("top_disease"),
        "top_probability": case.get("top_probability"),
        "detected_diseases": case.get("detected_diseases", []),
        "probabilities": case.get("probabilities", {}),
        "model_used": prediction.get("Model_Used"),
        "model_key": prediction.get("Model_Key"),
        "is_placeholder": case.get("is_placeholder"),
        "report_draft_kr": edited_report,
        "quality_check": (case.get("agent_profile") or {}).get("quality_check", {}),
        "triage_assessment": (case.get("agent_profile") or {}).get("triage_assessment", {}),
        "anatomy_assessment": (case.get("agent_profile") or {}).get("anatomy_assessment", {}),
    }


def submit_feedback(case: dict, feedback_type: str, threshold: float, corrected_labels: list[str], comment: str, reviewer_id: str, edited_report: str) -> None:
    payload = {
        "case_id": case.get("case_id", "CXR-UNKNOWN"),
        "feedback_type": feedback_type,
        "original_top_disease": case.get("top_disease"),
        "corrected_labels": corrected_labels,
        "comment": comment.strip(),
        "reviewer_id": reviewer_id.strip() or None,
        "model_key": (case.get("prediction", {}) or {}).get("Model_Key"),
        "threshold": threshold,
        "prediction_summary": prediction_summary_payload(case, edited_report),
    }
    saved = call_feedback_api(payload)
    if saved:
        st.success(f"{saved.get('message', '피드백이 저장되었습니다')} 큐 ID: {saved.get('queue_id', '-')}")


def safe_preview_image(uploaded_lookup: Dict[str, bytes], filename: str) -> Image.Image | None:
    raw = uploaded_lookup.get(filename)
    if raw is None or filename.lower().endswith((".dcm", ".dicom")):
        return None
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


with st.sidebar:
    st.markdown("### 🧠 Agentic Workbench")
    st.caption("MedRAX식 multi-tool CXR 워크플로우")
    health = check_api_health()
    if health:
        loaded = health.get("loaded_models", [])
        st.success(f"FastAPI 연결 · 로드 모델 {len(loaded)}개")
    else:
        st.error("FastAPI 미연결")
    st.divider()
    model_label = st.radio("분석 모델", list(MODEL_OPTIONS.values()), index=0)
    model_key = [k for k, v in MODEL_OPTIONS.items() if v == model_label][0]
    threshold = st.slider("감지 임계값", 0.1, 0.9, 0.3, 0.05)
    st.divider()
    st.page_link("app.py", label="단일 이미지 기본 분석")
    st.page_link("pages/analysis_results.py", label="Result Analysis")
    st.page_link("pages/reliability_readiness.py", label="Reliability Readiness")

st.markdown(
    """
<div class="agent-header">
    <div class="eyebrow">MedRAX-inspired Runtime Agent</div>
    <h1>Agentic Case Workbench</h1>
    <p>
        기존 CXR-CAD의 학습 완료 모델과 Grad-CAM, 판독문 초안, 의료진 피드백 큐는 그대로 유지하면서,
        여러 장의 X-ray/DICOM을 한 번에 입력하고 이미지별 결과·검진 초안·피드백·품질 점검·해부학 ROI·비교 요약을 제공하는 케이스 워크벤치입니다.
    </p>
    <span class="agent-pill">Multi-image upload</span>
    <span class="agent-pill">DICOM-aware routing</span>
    <span class="agent-pill">Tool trace</span>
    <span class="agent-pill">Per-case draft & feedback</span>
    <span class="agent-pill">Follow-up comparison</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("이 페이지가 기존 App과 다른 점", expanded=True):
    st.markdown(
        """
- **기존 App**: 1장 중심 분석, Grad-CAM, 판독문 초안, 의료진 피드백을 안정적으로 유지합니다.
- **Agentic Workbench**: 여러 장을 한 케이스 묶음으로 분석하고, MedRAX처럼 도구 호출 흐름을 보여주며, 이미지별 결과와 전체 비교를 함께 제공합니다.
- **학습 과정 변경 없음**: 새 모델을 학습하지 않고 기존 `/predict`, Grad-CAM, report draft, feedback queue를 Agent 오케스트레이션으로 확장합니다.
        """
    )

uploaded_files = st.file_uploader(
    "X-ray 또는 DICOM 파일을 여러 장 업로드하세요",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    accept_multiple_files=True,
    help="동일 환자의 과거/현재 영상 또는 여러 케이스를 한 번에 넣어 이미지별 초안과 비교 요약을 생성합니다.",
)
question = st.text_area(
    "Agent에게 물어볼 질문 또는 비교 요청",
    placeholder="예: 첫 번째 영상과 마지막 영상에서 악화된 소견이 있는지 비교해줘. / 기흉 의심 케이스를 우선순위로 정리해줘.",
    height=80,
)

run = st.button("Agent 분석 실행", type="primary", use_container_width=True, disabled=not uploaded_files or not health)
if run:
    with st.spinner("Agent가 입력 라우팅 → 모델 추론 → 판독 초안 → 품질 점검 → ROI 스캐폴드 → 비교 요약을 실행 중입니다..."):
        st.session_state["agent_result"] = call_agent_api(uploaded_files, model_key, threshold, question)
        st.session_state["agent_uploaded_lookup"] = {f.name: f.getvalue() for f in uploaded_files}

result = st.session_state.get("agent_result")
uploaded_lookup = st.session_state.get("agent_uploaded_lookup", {})

if not health:
    st.warning("백엔드가 연결되지 않았습니다. FastAPI를 실행한 뒤 다시 시도하세요.")
    st.code("uvicorn api.main:app --reload --port 8000", language="bash")
elif not result:
    st.markdown(
        """
<div class="agent-card">
    <h3>분석 대기 중</h3>
    <p>여러 장의 X-ray 또는 DICOM을 업로드한 뒤 Agent 분석을 실행하면, 이미지별 초안과 피드백 창구가 자동으로 생성됩니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    summary = result.get("agent_summary", {}) or {}
    cases = result.get("cases", [])

    st.markdown('<div class="section-title">Agent 전체 요약</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="agent-card">
    <h3>케이스 묶음 요약</h3>
    <p>{escape(str(summary.get('narrative', '')))}</p>
    <p class="small-muted">{escape(str(summary.get('safety_note', result.get('safety_note', ''))))}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><div class='value'>{result.get('case_count', 0)}</div><div class='label'>분석 이미지</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='value'>{summary.get('placeholder_count', 0)}</div><div class='label'>Placeholder</div></div>", unsafe_allow_html=True)
    with m3:
        comparison_on = "ON" if (summary.get("comparison") or {}).get("enabled") else "OFF"
        st.markdown(f"<div class='metric-card'><div class='value'>{comparison_on}</div><div class='label'>비교 분석</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><div class='value'>{result.get('model_key', '-')}</div><div class='label'>모델</div></div>", unsafe_allow_html=True)

    comparison = summary.get("comparison") or {}
    if comparison.get("enabled"):
        st.markdown('<div class="section-title">영상 간 변화 비교</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='agent-card'><p>{escape(str(comparison.get('summary', '')))}</p></div>", unsafe_allow_html=True)
        deltas = comparison.get("probability_deltas", [])
        if deltas:
            delta_df = pd.DataFrame(deltas)
            delta_df = delta_df.rename(columns={
                "label_kr": "질환",
                "first_probability": "첫 영상 확률",
                "last_probability": "마지막 영상 확률",
                "delta": "변화량",
            })[["질환", "첫 영상 확률", "마지막 영상 확률", "변화량"]]
            st.dataframe(delta_df, width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Agent Tool Trace</div>', unsafe_allow_html=True)
    trace_cols = st.columns(2)
    for idx, item in enumerate(result.get("tool_trace", [])):
        with trace_cols[idx % 2]:
            st.markdown(
                f"<div class='trace-step'><b>{item.get('step')}. {escape(str(item.get('tool')))}</b><br><span class='small-muted'>{escape(str(item.get('description')))}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">이미지별 판독·피드백</div>', unsafe_allow_html=True)
    if cases:
        tabs = st.tabs([f"{idx + 1}. {case.get('filename', 'case')}" for idx, case in enumerate(cases)])
        for tab, case in zip(tabs, cases):
            with tab:
                prediction = case.get("prediction", {}) or {}
                agent_profile = case.get("agent_profile", {}) or {}
                quality = agent_profile.get("quality_check", {}) or {}
                triage = agent_profile.get("triage_assessment", {}) or {}
                anatomy = agent_profile.get("anatomy_assessment", {}) or {}
                probs = case.get("probabilities", {}) or {}
                filename = case.get("filename", "uploaded")
                case_id = case.get("case_id", "CXR-UNKNOWN")
                triage_level = str(triage.get("triage_level", ""))
                triage_class = "demo" if case.get("is_placeholder") else ("urgent" if "URGENT" in triage_level else "high" if "HIGH" in triage_level else "normal")

                left, right = st.columns([1.15, 1.85], gap="large")
                with left:
                    preview = safe_preview_image(uploaded_lookup, filename)
                    st.markdown("#### 원본 이미지")
                    if preview is not None:
                        st.image(preview, width="stretch", caption=filename)
                    else:
                        st.info("DICOM 또는 미리보기 불가 파일입니다. Backend Agent가 DICOM 변환 후 분석했습니다.")
                    gradcam = prediction.get("GradCAM_Base64", "")
                    if gradcam and len(gradcam) > 500:
                        try:
                            st.markdown("#### Grad-CAM")
                            st.image(base64.b64decode(gradcam), width="stretch")
                        except Exception:
                            st.warning("Grad-CAM 이미지를 렌더링하지 못했습니다.")
                    else:
                        st.caption("실제 모델 Grad-CAM이 없거나 Placeholder 응답입니다.")

                with right:
                    st.markdown(
                        f"""
<div class="triage {triage_class}">
    <h4>{escape(str(triage.get('triage_label_kr', 'Agent 판정')))} · {escape(str(case.get('top_disease', '-')).replace('_', ' '))} {float(case.get('top_probability', 0.0)):.1%}</h4>
    <p>{escape(str(triage.get('reason', '')))}</p>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    tag_html = " ".join(
                        f"<span class='disease-tag'>{escape(DISEASE_LABELS_KR.get(d, d))} {float(probs.get(d, 0.0)):.0%}</span>"
                        for d in case.get("detected_diseases", [])
                    ) or "<span class='disease-tag'>임계값 이상 소견 없음</span>"
                    st.markdown(tag_html, unsafe_allow_html=True)
                    st.plotly_chart(render_bar_chart(probs, float(result.get("threshold", threshold))), width="stretch", config={"displayModeBar": False})

                st.markdown("#### 이미지별 AI 판독문 초안")
                edited_report = st.text_area(
                    "판독문 초안 편집",
                    value=case.get("report_draft", ""),
                    height=210,
                    key=f"agent_report_{case_id}",
                )
                st.download_button(
                    "이 이미지의 판독문 초안 다운로드",
                    data=edited_report.encode("utf-8"),
                    file_name=f"{case_id}_agent_report.txt",
                    mime="text/plain",
                    key=f"agent_download_{case_id}",
                )

                info_cols = st.columns(3)
                with info_cols[0]:
                    st.markdown("##### 품질 점검")
                    st.metric("품질 등급", quality.get("quality_grade", "-"), f"{quality.get('quality_score', '-')}/100")
                    for flag in quality.get("flags", [])[:3]:
                        st.caption(f"• {flag}")
                with info_cols[1]:
                    st.markdown("##### 해부학 ROI")
                    rois = anatomy.get("focus_rois", [])
                    if rois:
                        for roi in rois[:4]:
                            st.caption(f"• {roi.get('label_kr')} · {float(roi.get('priority_score', 0.0)):.0%}")
                    else:
                        st.caption("ROI 스캐폴드 없음")
                with info_cols[2]:
                    st.markdown("##### DICOM/메타데이터")
                    meta = prediction.get("Image_Metadata", {}) or {}
                    st.caption(f"크기: {meta.get('width', '-') } × {meta.get('height', '-')}")
                    st.caption(f"DICOM 입력: {'예' if meta.get('is_dicom_input') else '아니오'}")
                    if meta.get("dicom_metadata"):
                        with st.expander("DICOM 메타데이터 보기"):
                            st.json(meta.get("dicom_metadata"))

                with st.expander("해부학 ROI 상세 보기"):
                    st.caption(anatomy.get("disclaimer", ""))
                    roi_rows = []
                    for roi in anatomy.get("focus_rois", []):
                        roi_rows.append({
                            "ROI": roi.get("label_kr"),
                            "관련 소견": ", ".join(DISEASE_LABELS_KR.get(d, d) for d in roi.get("related_findings", [])),
                            "우선도": roi.get("priority_score"),
                            "검토 힌트": roi.get("review_hint"),
                            "bbox_px": roi.get("bbox_px"),
                        })
                    if roi_rows:
                        st.dataframe(pd.DataFrame(roi_rows), width="stretch", hide_index=True)

                st.markdown("#### 이 이미지에 대한 의료진 피드백")
                fb_left, fb_right = st.columns([1, 1])
                with fb_left:
                    reviewer_id = st.text_input("판독의/검수자 ID", key=f"agent_reviewer_{case_id}", placeholder="예: RAD01")
                    corrected_labels = st.multiselect(
                        "수정 라벨",
                        DISEASE_LABELS,
                        format_func=lambda x: f"{DISEASE_LABELS_KR.get(x, x)} / {x.replace('_', ' ')}",
                        key=f"agent_corrected_{case_id}",
                    )
                with fb_right:
                    comment = st.text_area("판독의 코멘트", key=f"agent_comment_{case_id}", height=105)

                fb_cols = st.columns(5)
                for fb_col, feedback_type in zip(fb_cols, FEEDBACK_TYPES):
                    with fb_col:
                        if st.button(feedback_type, key=f"agent_fb_{feedback_type}_{case_id}", use_container_width=True):
                            if feedback_type == "질환 라벨 수정" and not corrected_labels:
                                st.warning("라벨 수정 피드백에는 수정 라벨이 필요합니다.")
                            elif feedback_type == "판독의 코멘트" and not comment.strip():
                                st.warning("코멘트를 입력해야 저장할 수 있습니다.")
                            else:
                                submit_feedback(case, feedback_type, float(result.get("threshold", threshold)), corrected_labels, comment, reviewer_id, edited_report)
