"""
FastAPI 엔드포인트 유닛 테스트.

- GET  /health  응답 검증
- GET  /models  응답 검증
- POST /predict 응답 검증 (모델 선택 포함)
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _make_fake_png() -> bytes:
    """1×1 흰색 PNG 이미지 생성."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:

    def test_health_returns_200(self):
        """GET /health → 200 OK."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self):
        """GET /health 응답에 필수 필드 포함."""
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestModelsEndpoint:

    def test_models_returns_200(self):
        """GET /models → 200 OK."""
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_models_response_contains_all_keys(self):
        """GET /models 응답에 3개 모델 키 포함."""
        resp = client.get("/models")
        data = resp.json()
        assert "models" in data
        for key in ("densenet", "efficientnet", "vit"):
            assert key in data["models"], f"'{key}' 모델 정보 없음"


class TestPredictEndpoint:

    def test_predict_densenet_returns_200(self):
        """POST /predict?model=densenet → 200 OK."""
        png = _make_fake_png()
        resp = client.post(
            "/predict?model=densenet",
            files={"file": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200

    def test_predict_efficientnet_returns_200(self):
        """POST /predict?model=efficientnet → 200 OK."""
        png = _make_fake_png()
        resp = client.post(
            "/predict?model=efficientnet",
            files={"file": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200

    def test_predict_vit_returns_200(self):
        """POST /predict?model=vit → 200 OK."""
        png = _make_fake_png()
        resp = client.post(
            "/predict?model=vit",
            files={"file": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self):
        """POST /predict 응답에 14개 질환 확률 + 필수 필드 포함."""
        png = _make_fake_png()
        resp = client.post(
            "/predict?model=densenet",
            files={"file": ("test.png", png, "image/png")},
        )
        data = resp.json()
        disease_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia",
        ]
        for label in disease_labels:
            assert label in data, f"'{label}' 필드 없음"
            assert 0.0 <= data[label] <= 1.0, f"'{label}' 확률이 [0,1] 범위 벗어남"
        assert "Detected_Diseases" in data
        assert "Top_Disease" in data
        assert "Model_Used" in data
        assert "Model_Key" in data
        assert "Is_Placeholder" in data
        assert isinstance(data["Is_Placeholder"], bool)
        assert data["Model_Key"] == "densenet"

    def test_predict_invalid_model_returns_400(self):
        """POST /predict?model=invalid → 400 Bad Request."""
        png = _make_fake_png()
        resp = client.post(
            "/predict?model=resnet_invalid",
            files={"file": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 400

    def test_predict_empty_file_returns_400(self):
        """빈 파일 업로드 시 → 400 Bad Request."""
        resp = client.post(
            "/predict?model=densenet",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
