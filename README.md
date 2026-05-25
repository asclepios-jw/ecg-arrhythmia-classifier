# 🩺 ECG 심전도 부정맥 분류 AI

MIT-BIH Arrhythmia Dataset으로 학습한 1D CNN 기반 심박 분류 모델입니다.  
데이터 불균형 문제를 SMOTE + 클래스 가중치로 해결하고, Early Stopping으로 최적 모델을 자동 저장합니다.

---

## 📊 성능 결과

| 모델 | Accuracy | Macro F1 | Fusion F1 |
|------|----------|----------|-----------|
| 기본 모델 | 0.98 | 0.90 | 0.80 |
| 업그레이드 | 0.91 | 0.73 | 0.30 |
| 황금 밸런스 | 0.97 | 0.85 | 0.59 |
| **개선판 (최종)** | **0.98** | **0.91** | **0.79** |

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

---

## 🗂 클래스 정의

MIT-BIH 데이터셋은 5가지 심박 유형으로 구성됩니다.

| 라벨 | 클래스명 | 의미 |
|------|----------|------|
| N | Normal | 정상 박동 |
| S | Supraventricular | 상심실성 이상 박동 |
| V | Ventricular | 심실성 이상 박동 |
| F | Fusion | 정상 + 심실 박동이 동시에 발생 |
| Q | Unclassifiable | 분류 불가 박동 |

---

## 🔧 주요 개선 사항

- ✅ **SMOTE 오버샘플링** — 클래스 불균형 해소 (Fusion 0.7% → 균등 분포)
- ✅ **클래스 가중치 손실함수** — 희귀 클래스 오분류 시 추가 패널티
- ✅ **Early Stopping** — 검증 손실 기준 최적 모델 자동 저장
- ✅ **BatchNorm + Dropout** — 과적합 방지
- ✅ **ROC-AUC 평가** — Accuracy 외 의료 지표 추가
- ✅ **학습 곡선 시각화** — Train / Val Loss 추적

---

## 📁 프로젝트 구조

```
ecg-arrhythmia-classifier/
│
├── ecg_improved.ipynb       # 개선판 메인 노트북
├── ecg.ipynb                # 초기 실험 노트북 (Gemini 협업)
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🚀 실행 방법

### 1. 환경 설치

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn imbalanced-learn
```

### 2. 데이터 준비

[Kaggle — MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)에서 다운로드 후 아래 경로에 배치합니다.

```
/workspace/medical_datasets/share/archive/
├── mitbih_train.csv
└── mitbih_test.csv
```

### 3. 노트북 실행

```bash
jupyter notebook ecg_improved.ipynb
```

순서대로 셀을 실행하면 됩니다.  
GPU 환경을 권장하나, CPU에서도 동작합니다.

---

## 🧠 모델 아키텍처

```
입력 (1채널, 186 샘플)
    │
Conv1d(1→32) + BatchNorm + ReLU + MaxPool
    │
Conv1d(32→64) + BatchNorm + ReLU + MaxPool
    │
Linear(2944→128) + ReLU + Dropout(0.3)
    │
Linear(128→5)
    │
출력 (5가지 클래스)
```

---

## ⚠️ 주의사항

이 프로젝트는 **학습 및 연구 목적**으로 제작되었습니다.  
실제 임상 진단에 사용할 수 없으며, 의료 전문가의 판단을 대체하지 않습니다.

---

## 🛠 사용 스택

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)

---

## 📄 License

MIT License — 자세한 내용은 [LICENSE](./LICENSE)를 확인하세요.
