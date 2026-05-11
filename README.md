# On-device AI 기반 건설현장 안전장비 착용 감지 시스템

> Edge AI PPE (Personal Protective Equipment) Detection System  
> Raspberry Pi 5 + Hailo AI HAT | YOLOv8n + YOLOv8n-pose Ensemble

---

## 개요

클라우드 서버 없이 Edge Device에서 독립적으로 동작하는 건설현장 안전장비(PPE) 착용 감지 시스템입니다.

기존 PPE 탐지 시스템의 한계인 **착용/휴대 구분 불가** 문제를 YOLOv8n Detection + YOLOv8n-pose Ensemble로 해결하며, Raspberry Pi 5 + Hailo AI HAT에서 실시간으로 동작합니다.

---

## 주요 특징

- **이기종 모델 앙상블** — Detection + Pose Estimation 결합으로 착용/휴대/미착용 3단계 판단
- **On-device 완결 구조** — 영상이 외부로 나가지 않아 보안 확보, 네트워크 없이도 독립 동작
- **적응형 추론** — N프레임마다 Pose 실행으로 FPS 손실 최소화
- **위반 알림** — 지속 시간 기반 확정 + 로컬 로그 + GPIO 부저 경고

---

## 하드웨어 구성

| 부품 | 사양 |
|------|------|
| SBC | Raspberry Pi 5 8GB |
| AI 가속기 | Hailo AI HAT (Hailo-8L, 13 TOPS) |
| 카메라 | Pi Camera Module V2 |
| 경고 장치 | 부저/스피커 (GPIO) |

---

## 모델 성능

### Detection (YOLOv8n)

| Metric | YOLOv8n | YOLOv10n |
|--------|---------|----------|
| mAP@50 | 0.8192 | 0.7977 |
| mAP@50-95 | 0.4022 | 0.3937 |
| Precision | 0.8130 | 0.7801 |
| Recall | 0.7723 | 0.7679 |

### 클래스별 AP@50 (YOLOv8n)

| Class | AP@50 |
|-------|-------|
| helmet | 0.8927 |
| person | 0.8646 |
| vest | 0.7002 |

> 동일 데이터셋, 동일 하이퍼파라미터 기준 / YOLOv8n을 메인 모델로 채택

---

## 앙상블 판단 로직

```
카메라 입력
    ↓
YOLOv8n Detection (매 프레임)
    → person / helmet / vest bbox 추출
    ↓
YOLOv8n-pose (N프레임마다)
    → 머리 keypoint (코/눈/귀) 추출
    → 어깨 keypoint 추출
    ↓
매칭 + 판단
    → helmet: 머리 keypoint 근처 → worn / 하체 → carried / 없음 → not_detected
    → vest:   어깨 keypoint 근처 → worn / 없음 → not_detected
    ↓
Violation 판정 (N초 지속 + 버퍼링)
    → 부저 경고 + 로컬 로그 기록
```

---

## 파일 구조

```
ppe_system/
├── models/
│   ├── best.pt              # YOLOv8n Detection 가중치
│   └── yolov8n-pose.pt      # YOLOv8n Pose 가중치
│   └── *.hef                # Hailo 변환 모델 (HW 도착 후 추가)
├── detector_v1.py           # 기본 앙상블 로직 (베이스라인)
├── detector_v2.py           # 개선 버전 (conf 분리, 필터, 알림 포함)
├── detector.py              # 실행에 사용할 버전 (v1 또는 v2 복사)
├── visualizer.py            # 시각화
├── main.py                  # 실행 진입점 (RPi / PC 공통)
├── dashboard.py             # 관리 PC Flask 서버 (선택사항)
└── logs/
    └── violations.jsonl     # 위반 이력 로컬 기록
```

---

## detector 버전 비교

| 기능 | v1 (베이스라인) | v2 (현재) |
|------|----------------|-----------|
| Detection conf | 전체 0.4 통일 | 클래스별 분리 (helmet 0.45 / person 0.40 / vest 0.30) |
| helmet 필터 | 없음 | keypoint 기반 오탐 제거 |
| Pose IoU 매칭 | 0.5 (엄격) | 0.35 (완화) |
| 위반 알림 | 없음 | 지속 시간 기반 확정 + 쿨다운 |
| 로컬 로그 | 없음 | .jsonl 기록 |
| 부저 | 없음 | RPi GPIO (PC는 콘솔 출력) |

---

## 설치 및 실행

### 의존성 설치

```bash
pip install ultralytics opencv-python requests
# RPi에서는
pip install ultralytics opencv-python-headless requests --break-system-packages
```

### 모델 준비

```
models/best.pt          ← 학습된 Detection 가중치
models/yolov8n-pose.pt  ← Ultralytics 자동 다운로드 또는 직접 복사
```

### 실행할 detector 버전 선택

```bash
# v2 사용 시
copy detector_v2.py detector.py   # Windows
cp detector_v2.py detector.py     # Linux/Mac
```

### 실행

```bash
python main.py
```

종료: `q` 키 또는 `Ctrl+C`

---

## 주요 파라미터 (detector_v2.py Config)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `DET_CONF['helmet']` | 0.45 | helmet detection confidence |
| `DET_CONF['vest']` | 0.30 | vest detection confidence |
| `POSE_EVERY_N` | 3 | N프레임마다 Pose 실행 |
| `POSE_IOU_MATCH` | 0.35 | person-pose 매칭 IoU 임계값 |
| `HELMET_DY_RATIO` | 0.15 | helmet 착용 판정 y 허용 범위 |
| `VEST_DY_RATIO` | 0.35 | vest 착용 판정 y 허용 범위 |
| `VIOLATION_BUFFER_N` | 5 | 버퍼 프레임 수 |
| `VIOLATION_BUFFER_M` | 3 | 위반 확정 최소 프레임 수 |
| `VIOLATION_CONFIRM_SEC` | 3.0 | 위반 지속 시간 확정 기준 (초) |
| `ALERT_COOLDOWN_SEC` | 10.0 | 중복 알림 방지 쿨다운 (초) |

---

## 학습 환경

| 항목 | 내용 |
|------|------|
| 데이터셋 | Roboflow Construction Site Safety (~5,000장) |
| 클래스 | person / helmet / vest |
| 학습 환경 | Google Colab (T4 GPU) |
| Optimizer | AdamW (lr=0.001) |
| Epochs | 150 (early stopping patience=30) |
| Export | ONNX opset=13 (Hailo DFC 호환) |

---

## Hailo 변환 (HW 도착 후)

```bash
# ONNX → HEF 변환 (Hailo DFC)
hailo compile --hw-arch hailo8l --calib-path ./calibration best.onnx

# main.py에서 YOLO() 호출 부분을 HailoRT API로 교체
```

---

## 한계점 및 향후 과제

| 항목 | 내용 |
|------|------|
| 밀집 환경 | person-equipment 오매칭 가능 → ByteTrack 트래커 추가로 개선 예정 |
| 휴대 탐지 | 손에 든 helmet 학습 데이터 부족 → 데이터 보강 예정 |
| 야간 환경 | Pi Camera V2 저조도 한계 |
| Hailo 실측 | 현재 수치는 T4 GPU 기준, HW 도착 후 재측정 필요 |

---

## 팀 구성

| 이름 | 역할 |
|------|------|
| 손민석 | 안전장비 착용·미착용 데이터셋 수집 및 BBox 어노테이션 |
| 박건우 | 헬멧 휴대(holding) 데이터셋 수집 및 BBox 어노테이션 |
| 조윤호 | 안전 조끼(vest) 데이터셋 수집 및 BBox 어노테이션 |
| 홍윤오 | 파이프라인 구성 및 학습 환경 구축, BBox 어노테이션 |
