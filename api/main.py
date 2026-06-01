"""
CXR-CAD Backend API Service.

FastAPI 엔드포인트:
  GET  /health   → 서비스 상태 확인 (모델 로드 여부 포함)
  GET  /models   → 지원 모델 목록
  POST /predict  → 흉부 X-ray 분석 (PNG/JPEG/DICOM)

가중치 로드 규칙:
  - 서버 시작 시 checkpoints/<model_key>/<model_key>_best.pth 자동 탐색
  - 파일이 존재하면 실제 모델 추론, 없으면 Placeholder 모드
  - .pth 파일은 절대 Git 저장소에 포함하지 않습니다 (.gitignore 참조)

체크포인트 저장 포맷 (Colab 학습 코드와 호환):
    torch.save({
        "epoch"              : epoch,
        "model_state_dict"   : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auroc"          : best_auroc,
    }, "checkpoints/<model_key>/<model_key>_best.pth")
"""

from __future__ import annotations

import io
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import HealthResponse, ModelInfoResponse, PredictionResult
from src.preprocess.dicom_utils import dicom_to_pil, is_dicom
from src.preprocess.transforms import preprocess_single_image
from src.train.models import (
    DISEASE_LABELS,
    SUPPORTED_MODELS,
    build_model,
    get_model_info,
)

# ── 설정 ─────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR      = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
DETECTION_THRESHOLD = 0.3
API_VERSION         = "0.2.0"
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder Grad-CAM (1×1 빨간 픽셀 PNG, Base64)
_FAKE_GRADCAM_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8BQDwAEgAF/"
    "poIuwwAAAABJRU5ErkJggg=="
)


# ── 모델 레지스트리 ───────────────────────────────────────────────────────────

# { model_key: nn.Module | None }  None = 체크포인트 없음 (Placeholder 모드)
_model_registry: Dict[str, Optional[object]] = {k: None for k in SUPPORTED_MODELS}


def _find_checkpoint(model_key: str) -> Optional[Path]:
    """
    checkpoints/<model_key>/ 서브디렉토리에서 <model_key>_best.pth 탐색.

    탐색 우선순위:
      1. checkpoints/<model_key>/<model_key>_best.pth  ← 신규 구조
      2. checkpoints/<model_key>/*.pth (glob)           ← 신규 구조 변형
      3. checkpoints/<model_key>_best.pth              ← 구버전 flat 구조 (하위 호환)
    ex) checkpoints/densenet/densenet_best.pth
    """
    if not CHECKPOINT_DIR.exists():
        return None
    # 1) 신규: 서브디렉토리 내 직접 매칭
    sub_direct = CHECKPOINT_DIR / model_key / f"{model_key}_best.pth"
    if sub_direct.exists():
        return sub_direct
    # 2) 신규: 서브디렉토리 내 glob
    sub_candidates = sorted((CHECKPOINT_DIR / model_key).glob(f"{model_key}*.pth"), reverse=True)
    if sub_candidates:
        return sub_candidates[0]
    # 3) 구버전 flat 구조 fallback
    direct = CHECKPOINT_DIR / f"{model_key}_best.pth"
    if direct.exists():
        return direct
    candidates = sorted(CHECKPOINT_DIR.glob(f"{model_key}*.pth"), reverse=True)
    return candidates[0] if candidates else None


def _load_checkpoint_weights(model_key: str, ckpt_path: Path) -> bool:
    """
    .pth 파일에서 모델 가중치를 로드합니다.

    지원 state_dict 키 포맷:
      - {"model_state_dict": ...}   ← Colab 학습 표준 포맷
      - {"state_dict": ...}
      - 직접 state_dict (dict of tensors)

    Returns:
        True: 로드 성공, False: 실패
    """
    try:
        model = build_model(model_key)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            raise ValueError("알 수 없는 체크포인트 포맷")

        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        _model_registry[model_key] = model

        val_auroc = ckpt.get("val_auroc", "N/A") if isinstance(ckpt, dict) else "N/A"
        print(f"  ✅ [{model_key}] {ckpt_path.name} 로드 완료 (val_auroc={val_auroc})")
        return True

    except Exception as e:
        print(f"  ⚠️  [{model_key}] 로드 실패: {e}")
        return False


# ── Lifespan (서버 시작/종료 훅) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 checkpoints/<model>/ 폴더의 .pth 파일을 자동으로 탐색·로드합니다.

    - .pth 없음   → Placeholder 모드 (시뮬레이션 예측값 반환)
    - .pth 있음   → 실제 모델 추론
    """
    print(f"\n🩺 CXR-CAD API v{API_VERSION} 시작")
    print(f"   Device    : {DEVICE}")
    print(f"   Checkpoint: {CHECKPOINT_DIR.resolve()}")
    print("   모델 가중치 탐색 중...")

    loaded_any = False
    for key in SUPPORTED_MODELS:
        ckpt = _find_checkpoint(key)
        if ckpt:
            if _load_checkpoint_weights(key, ckpt):
                loaded_any = True
        else:
            print(f"  ℹ️  [{key}] 체크포인트 없음 → Placeholder 모드")

    if not loaded_any:
        print("\n   ⚠️  모든 모델이 Placeholder 모드로 동작합니다.")
        print("   Colab에서 학습 후 .pth 파일을 checkpoints/<model>/ 에 저장하세요.\n")

    app.state.loaded_models = [k for k, v in _model_registry.items() if v is not None]
    yield
    print("🩺 CXR-CAD API 종료")


# ── FastAPI App ───────────────────────────────────────────────────────────────

API_MODELS = SUPPORTED_MODELS + ["ensemble"]

app = FastAPI(
    title="CXR-CAD API",
    description=(
        "흉부 X-ray 컴퓨터 보조 진단 API.\n\n"
        "**지원 모델**: Ensemble, DenseNet-121, EfficientNet-B4, ViT-B/16\n\n"
        "모델 가중치는 Colab 학습 후 `checkpoints/<model>/` 서브폴더에 `.pth` 파일로 배치합니다.\n"
        "가중치 파일이 없으면 Placeholder 모드로 동작합니다."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Placeholder 예측 (가중치 없을 때) ────────────────────────────────────────

_PLACEHOLDER_BASE: Dict[str, Dict[str, float]] = {
    "densenet": {
        "Atelectasis": 0.32, "Cardiomegaly": 0.85, "Effusion": 0.50,
        "Infiltration": 0.18, "Mass": 0.12, "Nodule": 0.08,
        "Pneumonia": 0.22, "Pneumothorax": 0.05, "Consolidation": 0.15,
        "Edema": 0.42, "Emphysema": 0.03, "Fibrosis": 0.07,
        "Pleural_Thickening": 0.11, "Hernia": 0.02,
    },
    "efficientnet": {
        "Atelectasis": 0.29, "Cardiomegaly": 0.88, "Effusion": 0.53,
        "Infiltration": 0.20, "Mass": 0.14, "Nodule": 0.09,
        "Pneumonia": 0.24, "Pneumothorax": 0.06, "Consolidation": 0.17,
        "Edema": 0.45, "Emphysema": 0.04, "Fibrosis": 0.08,
        "Pleural_Thickening": 0.12, "Hernia": 0.02,
    },
    "vit": {
        "Atelectasis": 0.31, "Cardiomegaly": 0.83, "Effusion": 0.48,
        "Infiltration": 0.22, "Mass": 0.11, "Nodule": 0.07,
        "Pneumonia": 0.21, "Pneumothorax": 0.04, "Consolidation": 0.14,
        "Edema": 0.40, "Emphysema": 0.03, "Fibrosis": 0.06,
        "Pleural_Thickening": 0.10, "Hernia": 0.01,
    },
}


def _placeholder_predict(model_key: str) -> Dict[str, float]:
    if model_key == "ensemble":
        base_probs = {d: 0.0 for d in _PLACEHOLDER_BASE["densenet"].keys()}
        for k in SUPPORTED_MODELS:
            for d, p in _PLACEHOLDER_BASE[k].items():
                base_probs[d] += p / len(SUPPORTED_MODELS)
        base = base_probs
    else:
        base = _PLACEHOLDER_BASE.get(model_key, _PLACEHOLDER_BASE["densenet"])
        
    return {
        d: round(min(1.0, max(0.0, p + random.uniform(-0.04, 0.04))), 4)
        for d, p in base.items()
    }


# ── 실제 모델 추론 ────────────────────────────────────────────────────────────

def _real_predict(model_key: str, image: Image.Image) -> Dict[str, float]:
    tensor = preprocess_single_image(image).to(DEVICE)
    with torch.no_grad():
        if model_key == "ensemble":
            loaded_models = [m for m in _model_registry.values() if m is not None]
            if not loaded_models:
                raise ValueError("Ensemble 추론을 위한 모델이 로드되지 않았습니다.")
            probs_sum = torch.zeros(len(DISEASE_LABELS), device=DEVICE)
            for model in loaded_models:
                logits = model(tensor).squeeze(0)
                probs_sum += torch.sigmoid(logits)
            probs = (probs_sum / len(loaded_models)).cpu().tolist()
        else:
            model  = _model_registry[model_key]
            logits = model(tensor).squeeze(0)
            probs  = torch.sigmoid(logits).cpu().tolist()   # logits → 확률
            
    return {d: round(float(p), 4) for d, p in zip(DISEASE_LABELS, probs)}


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """서비스 상태 및 모델 로드 여부 반환."""
    loaded: List[str] = app.state.loaded_models
    return HealthResponse(
        status="healthy",
        model_loaded=len(loaded) > 0,
        model_version=f"v{API_VERSION}-{('ensemble' if len(loaded) > 1 else loaded[0]) if loaded else 'placeholder'}",
        loaded_models=loaded,
        version=API_VERSION,
    )


@app.get("/models", response_model=ModelInfoResponse, tags=["System"])
async def list_models():
    """지원 모델 목록 및 로드 상태 반환."""
    info = get_model_info()
    for key in info:
        if key == "ensemble":
            info[key]["is_loaded"] = any(m is not None for m in _model_registry.values())
        else:
            info[key]["is_loaded"] = _model_registry.get(key) is not None
    return ModelInfoResponse(models=info)


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="흉부 X-ray (PNG/JPEG) 또는 DICOM (.dcm)"),
    model: str = Query(
        default="ensemble",
        description="사용할 모델: ensemble | densenet | efficientnet | vit",
    ),
    threshold: float = Query(
        default=DETECTION_THRESHOLD,
        ge=0.0, le=1.0,
        description="질환 감지 임계값 (기본 0.3)",
    ),
):
    """
    흉부 X-ray를 업로드하고 14개 질환 확률을 반환합니다.

    - **model**    : 사용할 모델 키
    - **threshold**: 이 값 이상의 확률을 '감지됨'으로 분류
    - DICOM(.dcm) 파일도 지원됩니다.

    Streamlit 등 프론트엔드는 이 엔드포인트에 이미지만 전송하면 됩니다.
    모델 로드·추론은 서버가 전담합니다.
    """
    model_key = model.lower().strip()
    if model_key not in API_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델: '{model}'. 지원 목록: {API_MODELS}",
        )

    # 파일 읽기
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="빈 파일이 업로드되었습니다.")

    # 이미지 파싱 (DICOM 또는 일반 이미지)
    filename = file.filename or ""
    try:
        if filename.lower().endswith((".dcm", ".dicom")) or is_dicom(io.BytesIO(contents)):
            # DICOM → PIL (임시 파일 경유)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                image = dicom_to_pil(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            allowed = ("image/png", "image/jpeg", "image/jpg")
            if file.content_type and file.content_type not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail=f"지원하지 않는 파일 형식: {file.content_type}. PNG/JPEG/DICOM을 사용하세요.",
                )
            image = Image.open(io.BytesIO(contents)).convert("RGB")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {e}")

    # 추론
    start_ms = time.time()

    is_placeholder = False
    if model_key == "ensemble":
        if any(m is not None for m in _model_registry.values()):
            probs = _real_predict(model_key, image)
        else:
            is_placeholder = True
    else:
        if _model_registry.get(model_key) is not None:
            probs = _real_predict(model_key, image)
        else:
            is_placeholder = True

    if is_placeholder:
        time.sleep(0.3)
        probs = _placeholder_predict(model_key)

    inference_ms = int((time.time() - start_ms) * 1000)

    # 결과
    detected    = [d for d, p in probs.items() if p >= threshold]
    top_disease = max(probs, key=probs.get)
    model_name  = get_model_info().get(model_key, {}).get("display_name", model_key)

    gradcam_b64 = _FAKE_GRADCAM_B64
    if not is_placeholder:
        try:
            import numpy as np
            import cv2
            from src.analysis.gradcam import GradCAM, get_target_layer, apply_heatmap_overlay, cam_to_base64
            
            grad_model_key = model_key
            if model_key == "ensemble":
                loaded_keys = [k for k, v in _model_registry.items() if v is not None]
                if loaded_keys:
                    grad_model_key = loaded_keys[0]
            
            target_model = _model_registry.get(grad_model_key)
            if target_model is not None:
                class_idx = DISEASE_LABELS.index(top_disease)
                tensor = preprocess_single_image(image).to(DEVICE)
                
                target_layer = get_target_layer(target_model)
                gcam = GradCAM(target_model, target_layer)
                cam = gcam.generate(tensor, class_idx, image_size=(image.height, image.width))
                gcam.remove_hooks()
                
                orig_img = np.array(image.convert("RGB"))
                if orig_img.shape[:2] != (image.height, image.width):
                    orig_img = cv2.resize(orig_img, (image.width, image.height))
                overlay = apply_heatmap_overlay(orig_img, cam)
                gradcam_b64 = cam_to_base64(overlay)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    return PredictionResult(
        **probs,
        Detected_Diseases  = detected,
        Top_Disease        = top_disease,
        Top_Probability    = round(probs[top_disease], 4),
        GradCAM_Base64     = gradcam_b64,
        Inference_Time_ms  = inference_ms,
        Model_Used         = model_name,
        Model_Key          = model_key,
        Is_Placeholder     = is_placeholder,
    )
