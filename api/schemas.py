"""
Pydantic Schemas for CXR-CAD API.

요청/응답 모델 정의. 모델 선택 필드 추가.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ── 지원 모델 타입 ────────────────────────────────────────────────────────────

ModelType = Literal["ensemble", "densenet", "efficientnet", "vit"]


# ── 응답 스키마 ───────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """
    /predict 엔드포인트 응답 스키마.

    14개 질환별 확률, 요약 정보, Grad-CAM, 사용된 모델 정보 포함.
    """

    # ── 질환별 확률 ─────────────────────────────────────────────────────────
    Atelectasis:       float = Field(..., ge=0.0, le=1.0)
    Cardiomegaly:      float = Field(..., ge=0.0, le=1.0)
    Effusion:          float = Field(..., ge=0.0, le=1.0)
    Infiltration:      float = Field(..., ge=0.0, le=1.0)
    Mass:              float = Field(..., ge=0.0, le=1.0)
    Nodule:            float = Field(..., ge=0.0, le=1.0)
    Pneumonia:         float = Field(..., ge=0.0, le=1.0)
    Pneumothorax:      float = Field(..., ge=0.0, le=1.0)
    Consolidation:     float = Field(..., ge=0.0, le=1.0)
    Edema:             float = Field(..., ge=0.0, le=1.0)
    Emphysema:         float = Field(..., ge=0.0, le=1.0)
    Fibrosis:          float = Field(..., ge=0.0, le=1.0)
    Pleural_Thickening: float = Field(..., ge=0.0, le=1.0)
    Hernia:            float = Field(..., ge=0.0, le=1.0)

    # ── 요약 ────────────────────────────────────────────────────────────────
    Detected_Diseases: List[str]  = Field(..., description="임계값 이상 질환 목록")
    Top_Disease:       str        = Field(..., description="가장 높은 확률의 질환")
    Top_Probability:   float      = Field(..., ge=0.0, le=1.0, description="Top1 질환의 예측 확률")
    GradCAM_Base64:    str        = Field(..., description="Grad-CAM 히트맵 (Base64 PNG)")
    Inference_Time_ms: int        = Field(..., ge=0, description="추론 시간 (ms)")

    # ── 모델 정보 ────────────────────────────────────────────────────────────
    Model_Used:        str        = Field(..., description="사용된 모델 이름 (예: DenseNet-121)")
    Model_Key:         str        = Field(..., description="모델 키 (ensemble / densenet / efficientnet / vit)")
    Is_Placeholder:    bool       = Field(..., description="True이면 실제 체크포인트 추론이 아니라 Placeholder 데모 응답")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Atelectasis": 0.32, "Cardiomegaly": 0.85, "Effusion": 0.50,
                "Infiltration": 0.18, "Mass": 0.12, "Nodule": 0.08,
                "Pneumonia": 0.22, "Pneumothorax": 0.05, "Consolidation": 0.15,
                "Edema": 0.42, "Emphysema": 0.03, "Fibrosis": 0.07,
                "Pleural_Thickening": 0.11, "Hernia": 0.02,
                "Detected_Diseases": ["Cardiomegaly", "Effusion", "Edema"],
                "Top_Disease": "Cardiomegaly",
                "Top_Probability": 0.85,
                "GradCAM_Base64": "iVBORw0KGgoAAAANSUhEUg...",
                "Inference_Time_ms": 312,
                "Model_Used": "DenseNet-121",
                "Model_Key": "densenet",
                "Is_Placeholder": True,
            }]
        }
    }


class HealthResponse(BaseModel):
    """GET /health 응답 스키마."""
    status:        str       = Field(..., description="서비스 상태 (healthy / degraded)")
    model_loaded:  bool      = Field(..., description="모델 로드 여부")
    model_version: str       = Field(..., description="API 버전 (예: v1.0.0-ensemble)")
    loaded_models: List[str] = Field(default_factory=list, description="로드된 모델 키 목록")
    version:       str       = Field(..., description="API 버전")


class ModelInfoResponse(BaseModel):
    """GET /models 응답 스키마 — 지원 모델 목록."""
    models: dict = Field(..., description="지원 모델 정보 딕셔너리")
