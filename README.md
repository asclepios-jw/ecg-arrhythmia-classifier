# ECG 심전도 부정맥 분류 AI

MIT-BIH Arrhythmia Dataset으로 학습한 1D CNN 기반 심박 분류 모델입니다.
데이터 불균형을 다루기 위해 클래스 가중치와 SMOTE를 시도했고, Early Stopping으로 최적 모델을 저장합니다.

**결론: 최종 모델의 검증 정확도는 0.98, macro-F1은 0.91입니다.** 다만 불균형 보정이 뚜렷한 성능 향상을 만들지는 못했습니다. 보정 없는 기본 모델도 정확도 0.98, macro-F1 0.90이었고, 가장 적은 F(융합 박동)의 F1은 오히려 기본 모델이 0.80으로 최종 모델 0.79와 비슷합니다. 이 수치는 `mitbih_train.csv`에서 나눈 검증 데이터 값이고, `mitbih_test.csv`로는 평가하지 않았습니다.

## 성능 결과

| 모델 | 불균형 처리 | Accuracy | Macro F1 | Fusion F1 |
|------|----------|----------|----------|-----------|
| 기본 모델 | 없음 | 0.98 | 0.90 | 0.80 |
| 업그레이드 | 역빈도 클래스 가중치 | 0.91 | 0.73 | 0.30 |
| 황금 밸런스 | 역빈도의 제곱근 가중치 | 0.97 | 0.85 | 0.59 |
| **개선판 (최종)** | SMOTE (가중치는 사실상 1.0) | **0.98** | **0.91** | **0.79** |

최종 모델 분류 리포트 (검증 17,511건, ROC-AUC OvR 0.9927):

```
              precision    recall  f1-score   support

      Normal       0.99      0.99      0.99     14494
   Supravent       0.83      0.83      0.83       445
    Ventricu       0.96      0.96      0.96      1158
      Fusion       0.80      0.77      0.79       128
   Unclassif       0.99      0.99      0.99      1286

    accuracy                           0.98     17511
   macro avg       0.91      0.91      0.91     17511
```

희귀 클래스 S, F의 정밀도와 재현율은 처리 방식에 따라 크게 움직였습니다.

| 모델 | S 정밀도 | S 재현율 | F 정밀도 | F 재현율 | macro 재현율 |
|---|---|---|---|---|---|
| 기본 모델 | 0.95 | 0.68 | 0.89 | 0.72 | 0.87 |
| 업그레이드 | 0.41 | 0.92 | 0.18 | 0.94 | 0.94 |
| 황금 밸런스 | 0.79 | 0.79 | 0.46 | 0.83 | 0.91 |
| 개선판 (최종) | 0.83 | 0.83 | 0.80 | 0.77 | 0.91 |

기본 모델 대비 최종 모델은 S 재현율이 0.68 → 0.83으로 올랐지만 S 정밀도는 0.95 → 0.83으로 내려갔습니다. 앞의 세 모델과 최종 모델은 검증 분할이 달라(층화 여부) 위 표의 작은 차이는 비교 근거로 충분하지 않습니다.

이 레포에는 별도 그림 파일이 없습니다. 혼동행렬과 학습 곡선은 각 노트북의 셀 출력에서 볼 수 있습니다.

## 문제와 데이터

심박 하나를 입력받아 5가지 유형 중 하나로 분류합니다.

MIT-BIH 데이터셋은 5가지 심박 유형으로 구성됩니다.

| 라벨 | 클래스명 | 의미 | 개수 (`mitbih_train.csv`) |
|------|----------|------|---|
| N | Normal | 정상 박동 | 72,471 (82.8%) |
| S | Supraventricular | 상심실성 이상 박동 | 2,223 (2.5%) |
| V | Ventricular | 심실성 이상 박동 | 5,788 (6.6%) |
| F | Fusion | 정상 + 심실 박동이 동시에 발생 | 641 (0.7%) |
| Q | Unclassifiable | 분류 불가 박동 | 6,431 (7.3%) |

전부 N이라고만 답해도 정확도가 약 83%이므로, 전체 정확도만으로는 성능을 판단하기 어렵습니다. 그래서 macro-F1과 클래스별 지표를 함께 봤습니다.

## 파이프라인

```
mitbih_train.csv 87,554건  →  층화 분할 8:2 (학습 70,043 / 검증 17,511)
   →  학습 데이터에만 SMOTE (289,885건, 클래스당 57,977건)
   →  1D CNN 학습 (Adam, Early Stopping)  →  검증 손실 최저 모델로 평가
```

입력은 CSV의 앞 186개 열을 그대로 씁니다(0~185번). 186번 열은 쓰지 않고, 187번 열이 라벨입니다.

### 모델 아키텍처

```
입력 (1채널, 186 샘플)
    │
Conv1d(1→32, k=5) + BatchNorm + ReLU + MaxPool
    │
Conv1d(32→64, k=5) + BatchNorm + ReLU + MaxPool
    │
Linear(2944→128) + ReLU + Dropout(0.3)
    │
Linear(128→5)
    │
출력 (5가지 클래스)
```

### 학습 설정 (최종 모델)

| 항목 | 값 |
|---|---|
| 손실 | CrossEntropy (`balanced` 클래스 가중치, SMOTE 후 전부 1.0) |
| 최적화 | Adam lr=0.001, StepLR (5 epoch마다 0.5배) |
| 배치 | 512 |
| 학습량 | 최대 30 epoch, patience 5. 실제로는 11 epoch에서 종료 |
| 선택 모델 | 검증 손실 최저인 6 epoch (0.0857) |
| 환경 | Python 3.10, PyTorch 2.12, RTX 2070 |

## 주요 개선 사항

- **SMOTE 오버샘플링**: 학습 데이터의 클래스를 균등하게 맞췄습니다. 검증 데이터에는 적용하지 않습니다.
- **클래스 가중치 손실함수**: 코드에는 있지만, SMOTE 후 클래스가 균등해져 가중치가 모두 1.0입니다. 최종 모델에서는 효과가 없습니다.
- **Early Stopping**: 검증 손실 기준으로 최적 모델을 저장합니다.
- **BatchNorm + Dropout**: 과적합을 줄이기 위해 넣었습니다. 그래도 6 epoch 이후 검증 손실은 다시 올라갑니다.
- **ROC-AUC 평가**: Accuracy 외 지표를 추가했습니다.
- **학습 곡선 시각화**: Train / Val Loss를 추적합니다.

## 시행착오

| 단계 | 시도 | 결과 | 판단 |
|---|---|---|---|
| 1 | 합성 사인파 10,000개로 이진 CNN 연습 | 손실 0.0000 | 구조 확인용입니다. 쉬운 합성 데이터라 성능 근거가 아닙니다 |
| 2 | 실제 데이터, 불균형 처리 없음 | 정확도 0.98, S 재현율 0.68 | 전체 정확도는 높지만 S를 많이 놓침 |
| 3 | 역빈도 가중치 (F 28.24배) | 정확도 0.91, F 정밀도 0.18 | 재현율은 올랐지만 정상 박동을 희귀 클래스로 잘못 보내는 오류가 늘어남 |
| 4 | 제곱근 가중치 (F 5.31배) | 정확도 0.97, F 정밀도 0.46 | 중간 지점이지만 기본 모델의 macro-F1 0.90을 넘지 못함 |
| 5 | SMOTE + Early Stopping + StepLR | 정확도 0.98, macro-F1 0.91 | S 지표의 균형은 나아졌지만 전체 차이는 작음 |

처음에는 최종 모델을 "SMOTE와 클래스 가중치의 결합"으로 설명했습니다. 노트북 출력을 확인해 보면 SMOTE 후 가중치는 모두 1.0이어서, 실제로 작동한 것은 SMOTE입니다.

같은 구조를 다른 수업 과제용 스크립트로 다시 실행한 기록에서는 정확도 0.98, ROC-AUC 0.9917, F 재현율 0.80, F 정밀도 0.61이 나왔습니다. torch seed를 고정하지 않았기 때문에 희귀 클래스 지표는 실행마다 꽤 달라질 수 있습니다.

## 한계

- **시험 데이터로 평가하지 않았습니다.** `mitbih_test.csv`를 쓰지 않았고, 최종 수치를 낸 검증 데이터로 Early Stopping과 모델 선택까지 했습니다. 수치가 낙관적일 수 있습니다.
- **환자 단위로 나누지 않았습니다.** 심박 단위 무작위 분할이고 파일에 환자 정보가 없어서, 같은 환자의 심박이 학습과 검증에 함께 들어갔을 수 있습니다. 처음 보는 환자에 대한 성능이 아닙니다.
- **단일 실행 결과입니다.** torch seed를 고정하지 않았고, 반복 실행이나 신뢰구간이 없습니다. F는 검증 데이터에 128건뿐이라 지표가 쉽게 흔들립니다.
- **불균형 보정의 효과를 분리하지 못했습니다.** 최종 모델은 분할, SMOTE, Early Stopping, 스케줄러를 함께 바꿨습니다.
- **SMOTE의 한계가 있습니다.** 소수 클래스 심박 사이를 보간한 합성 파형이 실제 부정맥 파형과 같다는 보장은 없습니다.
- **가중치 파일을 포함하지 않았습니다.** `.pth`가 없어 결과를 재현하려면 다시 학습해야 합니다.

## 프로젝트 구조

```
ecg-arrhythmia-classifier/
│
├── ecg_improved.ipynb       # 개선판 메인 노트북
├── ecg.ipynb                # 초기 실험 노트북 (Gemini 협업): 기본, 업그레이드, 황금 밸런스
├── README.md
├── .gitignore
├── gitignore
└── LICENSE
```

## 실행 방법

### 1. 환경 설치

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn imbalanced-learn
```

### 2. 데이터 준비

[Kaggle — MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)에서 다운로드 후 아래 경로에 배치합니다. 다른 위치에 두려면 노트북의 경로(`real_data_path`, `best_path` 등)를 바꿔야 합니다.

```
/workspace/medical_datasets/share/archive/
├── mitbih_train.csv
└── mitbih_test.csv
```

### 3. 노트북 실행

```bash
jupyter notebook ecg_improved.ipynb
```

순서대로 셀을 실행하면 됩니다. 코드가 `.cuda()`를 직접 호출하므로 **GPU가 필요합니다.** CPU에서 돌리려면 해당 부분을 고쳐야 합니다. 노트북 첫 셀의 설치 목록에는 이 분석에 쓰지 않는 패키지(monai, SimpleITK 등)도 들어 있습니다. seed를 고정하지 않아 숫자는 위 표와 조금 다를 수 있습니다.

## 주의사항

이 프로젝트는 **학습 및 연구 목적**으로 제작되었습니다.
실제 임상 진단에 사용할 수 없으며, 의료 전문가의 판단을 대체하지 않습니다.

## 관련 레포

- [ecg-nearest-centroid](https://github.com/asclepios-jw/ecg-nearest-centroid) — 같은 데이터로 numpy만 써서 만든 최근접 중심점 기준선 (`mitbih_test.csv` 정확도 45.6%). 평가 데이터가 달라 이 레포의 수치와 직접 비교할 수는 없습니다.

## 사용 스택

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)

## License

MIT License — 자세한 내용은 [LICENSE](./LICENSE)를 확인하세요.
