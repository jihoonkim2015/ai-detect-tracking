# AI 영상 기반 객체인식 및 트래킹 시스템

파이토치 기반의 실시간 객체 감지 및 트래킹 시스템입니다. YOLOv8과 DeepSORT 알고리즘을 활용하여 비디오나 웹캠에서 객체를 감지하고 추적합니다.

## 주요 기능

- **실시간 객체 감지**: YOLOv8을 사용한 고성능 객체 감지
- **객체 트래킹**: DeepSORT 알고리즘을 통한 안정적인 객체 추적
- **다중 입력 지원**: 웹캠, 비디오 파일, 이미지 처리 지원
- **실시간 시각화**: 감지된 객체와 트래킹 경로 실시간 표시
- **성능 모니터링**: FPS 및 처리 시간 실시간 모니터링

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 웹캠으로 실시간 감지 및 트래킹
```bash
python main.py --source 0 --save-txt --save-conf
```

### 비디오 파일 처리
```bash
python main.py --source path/to/video.mp4 --save-txt --save-conf
```

### 이미지 처리
```bash
python main.py --source path/to/image.jpg --save-txt --save-conf
```

## 프로젝트 구조

```
ai-detect-tracking/
├── main.py                 # 메인 실행 파일
├── models/
│   ├── __init__.py
│   ├── detector.py         # 객체 감지 모델
│   └── tracker.py          # 객체 트래킹 모델
├── utils/
│   ├── __init__.py
│   ├── video_utils.py      # 비디오 처리 유틸리티
│   ├── visualization.py    # 시각화 유틸리티
│   └── config.py          # 설정 파일
├── data/
│   ├── videos/            # 테스트 비디오 파일
│   └── outputs/           # 처리 결과 저장
├── requirements.txt
└── README.md
```

## 설정

`utils/config.py`에서 다양한 설정을 조정할 수 있습니다:
- 모델 설정 (confidence threshold, IoU threshold)
- 트래킹 설정 (max_age, min_hits)
- 출력 설정 (저장 경로, 파일 형식)

## 지원 객체 클래스

COCO 데이터셋의 80개 클래스를 지원합니다:
- 사람 (person)
- 차량 (car, bus, truck, bicycle, motorcycle)
- 동물 (dog, cat, bird, horse, cow 등)
- 일상용품 등

## 라이센스

MIT License
