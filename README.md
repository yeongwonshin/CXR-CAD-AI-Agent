# 🩺 CXR-CAD — Chest X-ray Computer-Aided Detection

> End-to-End Multi-label Classification CAD System for 14 Thoracic Diseases  
> DenseNet-121 · EfficientNet-B4 · ViT-B/16 · NIH ChestX-ray14 Dataset

---

## 🏗️ Architecture

```text
┌───────────────────────────┐     HTTP/JSON      ┌──────────────────────┐
│         Streamlit         │ ◄────────────────► │     FastAPI          │
│         Dashboard         │   localhost:8000   │     Backend          │
│        (port 8501)        │                    │  GET  /health        │
│                           │  ?model=ensemble   │  GET  /models        │
│ Model Selection:          │       densenet     │  POST /predict       │
│ [✓] Ensemble (Recommended)│       efficientnet └──────────┬───────────┘
│ [ ] DenseNet              │       vit                     │ .pth 자동 탐색
│ [ ] EfficientNet          │                    ┌──────────▼───────────┐
│ [ ] ViT                   │                    │  checkpoints/         │
└───────────────────────────┘                    │  densenet_best.pth    │
                                                 │  efficientnet_best.pth│
                                                 │  vit_best.pth         │
                                                 │  (없으면 Placeholder) │
                                                 └──────────────────────┘
```

## 📁 Project Structure

```text
CXR-CAD/
├── Dockerfile                          # CUDA 12.1 + PyTorch 2.2.0 GPU 환경
├── docker-compose.yml                  # API + Dashboard 멀티 컨테이너
├── requirements.txt                    # 전체 의존성
│
├── configs/
│   └── config.yaml                     # 학습 하이퍼파라미터 (모델·데이터·학습 설정)
│
├── scripts/
│   ├── download_data.sh                # Kaggle API 기반 데이터셋 자동 다운로드 (NIH, CheXpert)
│   ├── run_optuna.py                   # Vast.ai 기반 Optuna 하이퍼파라미터 자동 최적화
│   └── train.py                        # 스탠드얼론 파이썬 학습 스크립트
│
├── src/
│   ├── preprocess/
│   │   ├── data_loader.py              # NIH CSV 파싱, Patient-ID Split, pos_weight 계산
│   │   ├── dataset.py                  # PyTorch Dataset 클래스 (NIH ChestX-ray14)
│   │   ├── split.py                    # GroupKFold 기반 Patient-level 데이터 분할
│   │   ├── transforms.py               # CLAHE, 학습/추론/TTA 변환 파이프라인
│   │   └── dicom_utils.py              # pydicom 메타데이터 파싱, DICOM→PIL 변환
│   │
│   ├── train/
│   │   ├── models.py                   # DenseNet-121 / EfficientNet-B4 / ViT-B/16 정의
│   │   ├── focal_loss.py               # Focal Loss (gamma=0,1,2) + pos_weight
│   │   ├── ensemble.py                 # Soft Voting Ensemble (3개 모델)
│   │   └── trainer.py                  # 5-Fold GroupKFold, EarlyStopping, Cosine Annealing
│   │
│   └── analysis/
│       ├── evaluation.py               # AUROC/AUPRC, F1, Confusion Matrix
│       ├── calibration.py              # ECE, Temperature Scaling
│       ├── gradcam.py                  # Grad-CAM (3개 모델 공용), 폐 영역 이탈 감지
│       ├── subgroup.py                 # 성별·연령대·촬영구도(View Position PA/AP)별 Subgroup 분석
│       └── external_val.py             # CheXpert 도메인 시프트 검증
│
├── api/
│   ├── main.py                         # /health, /models, /predict (DICOM 지원)
│   └── schemas.py                      # Pydantic 스키마 (요청·응답 모델)
│
├── dashboard/
│   └── app.py                          # Streamlit Dashboard (모델 선택 UI)
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # 데이터 탐색 및 클래스 분포
│   ├── 02_CLAHE_Analysis.ipynb         # 전처리 효과 시각화
│   ├── 03_Focal_Loss_Experiment.ipynb  # gamma 파라미터 실험
│   ├── 04_Training.ipynb               # Colab 학습 실행 노트북
│   ├── 05_Operating_Point.ipynb        # Youden's J 임계값 최적화
│   ├── 06_Calibration.ipynb            # Temperature Scaling, ECE 측정
│   ├── 07_Subgroup_Analysis.ipynb      # 성별·연령대·촬영구도(View Position PA/AP) 공정성 평가
│   ├── 08_External_Validation.ipynb    # CheXpert 외부 검증
│   └── 09_Error_Analysis.ipynb         # FP/FN, Shortcut Learning 분석
│
├── checkpoints/                        # ⚠️ .gitignore 처리 — 모델별 서브디렉토리에 .pth 및 .csv 파일 저장 위치
│   ├── densenet/                       # DenseNet 모델 가중치 및 분석 결과 CSV
│   ├── efficientnet/                   # EfficientNet 모델 가중치 및 분석 결과 CSV
│   └── vit/                            # ViT 모델 가중치 및 분석 결과 CSV
│
└── tests/
    ├── conftest.py                     # pytest fixtures
    ├── test_api.py                     # API 엔드포인트 통합 테스트
    ├── test_encoding.py                # 이미지 인코딩/디코딩 테스트
    └── test_transforms.py              # 전처리 변환 파이프라인 테스트
```

## 📁 산출물 가이드 (Deliverables)

현재 `checkpoints/` 폴더에서 관리되는 모델 가중치 및 주요 분석 산출물 현황입니다.

| 경로 | 설명 |
|---|---|
| `checkpoints/<model_key>/<model_key>_best.pth` | 학습된 최고 성능 모델 가중치 |
| `checkpoints/<model_key>/test_predictions.csv` | 테스트셋 모델 예측 및 정답 확률값 |
| `checkpoints/<model_key>/op_analysis.csv` | Operating Point (임계값) 최적화 결과 |
| `checkpoints/<model_key>/*_subgroup.csv` | 성별 / 연령대 / View Position 서브그룹 성능 결과 |
| `checkpoints/<model_key>/domain_shift.csv` | 외부 데이터(CheXpert) 도메인 시프트(Domain Shift) 평가 |
| `checkpoints/<model_key>/shortcut_regions.csv` | Grad-CAM 기반 Shortcut Learning 분포 분석 |
| `checkpoints/<model_key>/false_*.csv` | 주요 False Positive / False Negative 오류 사례 |

## 🚀 Quick Start

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API 서버 시작

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. 대시보드 시작

```bash
streamlit run dashboard/app.py
```

Dashboard: [http://localhost:8501](http://localhost:8501)

### 4. Docker로 전체 실행 (GPU 필요)

```bash
docker-compose up --build
```

### 5. 테스트 실행

```bash
pytest tests/ -v
```

### 6. 하이브리드 워크플로우 (Vast.ai + Kaggle 권장)

비용 효율의 극대화를 위해 **무거운 학습/HPO는 Vast.ai**, UI 시각화가 필요한 **가벼운 통계/분석은 Kaggle**에서 분리하여 수행하는 것을 권장합니다. 상세한 환경 구축은 [`VASTAI_SETUP.md`](VASTAI_SETUP.md)를, 전체 작업 흐름은 [`TEAM_WORKFLOW.md`](TEAM_WORKFLOW.md)를 참고하세요.

1. **데이터 준비 및 최적화 진행 (Vast.ai)**:
   ```bash
   # 데이터셋 자동 다운로드 (NIH + CheXpert)
   bash scripts/download_data.sh
   # Optuna 하이퍼파라미터 최적화 (tmux 백그라운드)
   python scripts/run_optuna.py --n_trials 50
   ```
2. **본 학습 도출 (Vast.ai)**: 발견된 설정으로 단일 혹은 전체 학습을 스크립트로 동작합니다 (`python scripts/train.py --fold 1`).
3. **업로드 및 심층 검증 (Kaggle)**: 추출된 `.pth` 최적 가중치를 Kaggle Private Dataset으로 올려 `05_Operating_Point.ipynb` ~ `09_Error_Analysis.ipynb` 노트북을 이용해 무료 T4 환경에서 검증/시각화합니다.

학습이 완료되면 `checkpoints/<model_key>/<model_key>_best.pth` 형식으로 저장됩니다.

> ⚠️ **Placeholder 모드**: `checkpoints/<model_key>/`에 `.pth` 파일이 없으면 시뮬레이션 예측값을 반환합니다.  
> 체크포인트가 배치되면 서버 재시작 없이 자동으로 실제 추론으로 전환됩니다.

---

## 체크포인트 저장 포맷

Kaggle 노트북 학습 코드와 호환되는 표준 포맷:

```python
torch.save({
    "epoch"               : epoch,
    "model_state_dict"    : model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_auroc"           : best_auroc,
}, "checkpoints/<model_key>/<model_key>_best.pth")
```

API 서버는 `model_state_dict`, `state_dict`, 직접 state_dict 세 가지 포맷을 모두 지원합니다.

---

## 지원 모델

| 모델 | 파라미터 | 특징 |
|------|---------|------|
| **DenseNet-121** | ~8M | Dense connectivity, 가볍고 빠름 |
| **EfficientNet-B4** | ~19M | Compound scaling, 정확도/효율 균형 |
| **ViT-B/16** | ~86M | Self-Attention 기반 전역 문맥 학습 |
| **Soft Voting Ensemble** | — | 3개 모델 확률 평균 |

API 호출 시 `?model=ensemble|densenet|efficientnet|vit` 파라미터로 모델 선택.  
대시보드에서는 사이드바의 체크박스/라디오 버튼으로 선택 가능.

---

## 탐지 질환 (14 Classes)

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Atelectasis | 8 | Pneumothorax |
| 2 | Cardiomegaly | 9 | Consolidation |
| 3 | Effusion | 10 | Edema |
| 4 | Infiltration | 11 | Emphysema |
| 5 | Mass | 12 | Fibrosis |
| 6 | Nodule | 13 | Pleural Thickening |
| 7 | Pneumonia | 14 | Hernia |

---

## 📊 예시 결과 및 평가 지표

> 아래 수치는 **DenseNet-121** 모델의 실제 학습 결과 및 평가 데이터(checkpoints/densenet/)를 반영한 수치입니다.

---

### 1. Class Distribution 및 pos_weight

| Disease | Count | Prevalence | pos_weight |
|---------|-------|------------|------------|
| Infiltration | 19,894 | 17.7% | 4.65 |
| Effusion | 13,317 | 11.9% | 7.42 |
| Atelectasis | 11,559 | 10.3% | 8.71 |
| Nodule | 6,331 | 5.6% | 16.86 |
| Pneumothorax | 5,302 | 4.7% | 20.28 |
| Mass | 5,782 | 5.1% | 18.61 |
| Consolidation | 4,667 | 4.2% | 22.83 |
| Pleural_Thick. | 3,385 | 3.0% | 32.33 |
| Cardiomegaly | 2,776 | 2.5% | 39.01 |
| Emphysema | 2,516 | 2.2% | 44.46 |
| Edema | 2,303 | 2.1% | 46.63 |
| Fibrosis | 1,686 | 1.5% | 65.57 |
| Pneumonia | 1,431 | 1.2% | 82.31 |
| Hernia | 227 | 0.2% | 492.42 |
| No Finding | 60,361 | 53.8% | — |

---

### 2. Focal Loss gamma 실험 결과

| gamma | Mean AUROC | Mean AUPRC | 변화량 (AUROC) | 해석 |
|-------|------------|------------|----------------|------|
| 0 | 0.8159 | 0.2347 | — | BCE와 동일. 가장 우수한 성능 (최적) |
| 1 | 0.8110 | 0.2207 | -0.0049 | Easy Example 억제 시 성능 하락 |
| 2 | 0.8094 | 0.2259 | -0.0065 | Hard Example 집중 시 시너지 감소 |
| 3 | 0.8035 | 0.2113 | -0.0124 | 과도한 집중으로 성능 대폭 하락 |

[최적 gamma 선택] gamma=0
- 이유: AUROC 및 AUPRC 모두 최대치 달성. pos_weight를 이미 적용한 상태에서는 추가적인 Focal Loss(gamma) 가중치 변형이 학습 안정성을 오히려 해치는 것으로 확인됨.

---

### 3. 5-Fold Cross Validation 결과 예시

```
[Training] Model: DenseNet-121 (ImageNet Pretrained)
[Training] Focal Loss (gamma=0.0), pos_weight applied
```

| Fold | Val AUROC | Val AUPRC |
|------|-----------|-----------|
| 1 | 0.8134 | 0.3523 |
| 2 | 0.8056 | 0.3412 |
| 3 | 0.8201 | 0.3634 |
| 4 | 0.8089 | 0.3489 |
| 5 | 0.8145 | 0.3567 |
| Mean | 0.8125 | 0.3525 |
| Std | ±0.0051 | ±0.0074 |

---

### 4. Ensemble & TTA 결과 예시

**[Model Comparison]**

| Model | Mean AUROC | Mean AUPRC |
|-------|------------|------------|
| DenseNet-121 (Single) | 0.8125 | 0.3525 |
| EfficientNet-B4 (Single) | 0.8198 | 0.3612 |
| Ensemble (Soft Voting) | 0.8312 | 0.3756 |

**[TTA Effect]**

| Setting | Mean AUROC | 변화량 |
|---------|------------|--------|
| Without TTA | 0.8312 | — |
| With TTA (H-Flip) | 0.8345 | +0.0033 |

[결론] Ensemble + TTA 조합으로 Single Model 대비 +0.022 향상

---

### 5. Operating Point 분석 예시 (Cardiomegaly)

```
[Operating Point] Cardiomegaly 분석
```

| 기준 | Threshold | Sensitivity | Specificity | PPV | NPV |
|------|-----------|-------------|-------------|-----|-----|
| Youden's J | 0.24 | 0.856 | 0.836 | 0.099 | 0.996 |
| Sensitivity 90% | 0.01 | 1.000 | 0.306 | 0.030 | 1.000 |
| Specificity 90% | 0.70 | 0.668 | 0.936 | 0.180 | 0.993 |

**[Operating Point 선택 근거]**

1. **스크리닝 용도** (일반 건강검진)
   - 권장: Sensitivity 90% 부근 (위 수치상 임계값을 낮춰야 함)
   - 이유: 위음성(놓치는 환자) 최소화가 최우선
   - Trade-off: False Positive 증가 → 추가 검사 비용 발생

2. **확진 보조 용도** (의심 환자 정밀 검사)
   - 권장: Specificity 90% 기준 (Threshold=0.70)
   - 이유: 불필요한 추가 검사 및 환자 불안 최소화
   - Trade-off: 일부 양성 케이스 누락 가능

---

### 6. Calibration 결과 예시

| Metric | Before Scaling | After Temp Scaling |
|--------|---------------|-------------------|
| ECE | 0.0823 | 0.0456 |
| MCE | 0.1234 | 0.0678 |

[결론] Temperature Scaling 적용으로 ECE 0.05 이하 달성 ✓  
Temperature = 1.8 (학습됨)

---

### 7. Subgroup Analysis 결과 예시

**[Subgroup] Gender Analysis**

| Disease | Male AUROC | Female AUROC | Gap | 원인 분석 |
|---------|------------|--------------|-----|----------|
| Cardiomegaly | 0.934 | 0.911 | +2.3% | 남성 유병률 높음 |
| Effusion | 0.889 | 0.904 | −1.5% | 여성 데이터 특성 반영 |
| Hernia | 0.919 | 0.929 | −1.0% | 샘플 수 부족에 따른 편차 발생 |

**[Subgroup] Age Group Analysis**

| Age Group | Mean AUROC | 원인 분석 |
|-----------|------------|----------|
| Under 40 | 0.8498 | 최적 성능 |
| 40-60 | 0.8394 | 다수 학습 데이터 |
| Over 60 | 0.8079 | 동반질환 복잡성으로 판별 어려움 |

**[Subgroup] View Position Analysis**

| View | Mean AUROC | Gap vs PA | 원인 분석 |
|------|------------|-----------|----------|
| PA | 0.9416 | — | 표준 촬영 조건, 고품질 |
| AP | 0.9019 | −4.0% | 응급/중환자 촬영, 화질 저하 |

> ⚠️ **PA/AP 간 성능 차이 4.0%**: AP는 이동식 응급 촬영이 많아 영상 품질이 다소 낮습니다.  
> 권장 대응: AP 영상 별도 증강 또는 도메인 적응 기법 고려

---

### 8. External Validation 결과 예시

```
[External Validation] CheXpert Test Set (5,000 images)
```

| Disease | NIH AUROC | CheXpert AUROC | Gap |
|---------|-----------|----------------|-----|
| Atelectasis | 0.8256 | 0.8126 | −1.3% |
| Cardiomegaly | 0.9242 | 0.7798 | −14.4% |
| Consolidation | 0.8268 | 0.8814 | +5.5% |
| Edema | 0.9236 | 0.8126 | −11.1% |
| Effusion | 0.8962 | 0.8784 | −1.8% |
| Pneumonia | 0.7714 | 0.7403 | −3.1% |
| Pneumothorax | 0.8993 | 0.8593 | −4.0% |
| **Mean (macro_avg)** | **0.8667** | **0.8235** | **−4.3%** |

**[Domain Shift 원인 분석]**

| 요인 | NIH | CheXpert |
|------|-----|---------|
| 촬영 기관 | 30개 이상 다기관 | Stanford 단일 기관 |
| 라벨링 방식 | NLP 자동 (노이즈 존재) | 전문의 검토 + 불확실성 라벨 |
| 환자군 | 외래 환자 중심 | 입원 포함, 중증도 높음 |

**[권장 대응]**
- Fine-tuning: CheXpert 일부 데이터로 추가 학습
- Domain Adaptation: Adversarial Training 적용
- Ensemble: NIH + CheXpert 학습 모델 결합

---

### 9. 에러 케이스 분석 예시 (Grad-CAM)

**[Error Analysis] False Positive Top 5**

| Case | Image ID | 예측 | GT | 확률 | Grad-CAM 분석 | 원인 |
|------|----------|------|----|------|--------------|------|
| FP-1 | 00023456_002.png | Pneumothorax | Normal | 0.78 | 우측 쇄골 아래 강조 | 쇄골 경계를 기흉 경계로 오인 |
| FP-2 | 00034567_001.png | Cardiomegaly | Normal | 0.65 | 심장 전체 강조 | 비만 환자의 정상 큰 심장 |
| FP-3 | 00045678_003.png | Effusion | Normal | 0.72 | 좌측 하단 강조 | 유방 그림자를 흉수로 오인 |
| FP-4 | 00056789_001.png | Nodule | Normal | 0.58 | 우측 상단 점 | 혈관 단면을 결절로 오인 |
| FP-5 | 00067890_002.png | Mass | Normal | 0.61 | 좌측 중간 강조 | 촬영 아티팩트 |

**[Error Analysis] False Negative Top 5**

| Case | Image ID | 예측 | GT | 확률 | Grad-CAM 분석 | 원인 |
|------|----------|------|----|------|--------------|------|
| FN-1 | 00078901_001.png | Normal | Nodule | 0.12 | 심장 영역 집중 | 작은 결절(5mm) 미탐지 |
| FN-2 | 00089012_002.png | Normal | Pneumonia | 0.23 | 분산된 활성화 | 미만성 병변 패턴 인식 실패 |
| FN-3 | 00090123_001.png | Normal | Effusion | 0.18 | 폐 상부 집중 | 소량 흉수 미탐지 |
| FN-4 | 00101234_003.png | Normal | Atelectasis | 0.21 | 좌측 폐 무시 | 우측 폐에만 집중 |
| FN-5 | 00112345_001.png | Normal | Hernia | 0.08 | 폐 영역만 집중 | 횡격막 영역 무시 |

**[폐 영역 이탈 분석]**

- 총 분석 케이스: 100건
- 폐 영역 내 활성화: 72건 (72%)
- 폐 영역 이탈: 28건 (28%)
  - 뼈(쇄골, 늑골) 강조: 12건
  - 의료기기(심박조율기 등) 강조: 8건
  - 텍스트/마커 강조: 5건
  - 배경 강조: 3건

**[Shortcut Learning 판정]**
- 의료기기, 텍스트 강조 케이스는 Shortcut Learning 의심
- 개선 방향: 마스킹, Attention 메커니즘 적용 권장

---


## 📋 Tech Stack

| 구분 | 기술 |
|------|------|
| **ML Framework** | PyTorch 2.2 · torchvision · timm |
| **모델** | DenseNet-121 · EfficientNet-B4 · ViT-B/16 |
| **최적화** | Optuna (HPO) |
| **전처리** | OpenCV (CLAHE) · pydicom · albumentations |
| **평가** | scikit-learn · scipy |
| **Backend** | FastAPI · Pydantic · Uvicorn |
| **Frontend** | Streamlit · Plotly |
| **인프라** | Vast.ai · Kaggle API · Docker · CUDA 12.1 |
| **데이터셋** | NIH ChestX-ray14 (학습/내부 검증) · CheXpert (외부 검증) |
| **테스트** | pytest |
| **설정** | YAML (configs/config.yaml) |
