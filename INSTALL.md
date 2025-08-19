# AI 영상 기반 객체인식 및 트래킹 시스템 설치 가이드

## 시스템 요구사항

### 1. Python 환경
- Python 3.8 이상
- pip 패키지 관리자

### 2. 하드웨어 요구사항
- CPU: Intel i5 이상 또는 AMD 동급
- RAM: 8GB 이상 (16GB 권장)
- GPU: NVIDIA GPU (CUDA 지원) - 선택사항이지만 성능 향상에 도움
- 웹캠 또는 비디오 파일

## 설치 방법

### 1단계: Python 설치
1. [Python 공식 웹사이트](https://www.python.org/downloads/)에서 Python 3.8 이상 다운로드
2. 설치 시 "Add Python to PATH" 옵션 체크
3. 설치 완료 후 PowerShell에서 `python --version` 명령으로 확인

### 2단계: 프로젝트 의존성 설치
```bash
# 프로젝트 디렉토리로 이동
cd ai-detect-tracking

# 자동 설치 스크립트 실행 (Windows)
install.bat

# 또는 수동 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3단계: 시스템 테스트
```bash
python test_system.py
```

## 사용법

### 기본 실행
```bash
# 웹캠으로 실시간 감지 및 트래킹
python main.py --source 0

# 비디오 파일 처리
python main.py --source path/to/video.mp4

# 이미지 파일 처리
python main.py --source path/to/image.jpg
```

### 고급 옵션
```bash
# 결과 저장과 함께 실행
python main.py --source 0 --save-txt --save-conf

# 신뢰도 임계값 조정
python main.py --source 0 --conf-thres 0.7

# 화면 출력 없이 처리
python main.py --source video.mp4 --no-display

# 데모 실행 (대화형 메뉴)
python demo.py
```

### Windows 사용자를 위한 간편 실행
```bash
# 실행 스크립트
run.bat
```

## 주요 기능

### 1. 실시간 객체 감지
- YOLOv8 모델 사용
- COCO 데이터셋 80개 클래스 지원
- 높은 정확도와 빠른 처리 속도

### 2. 객체 트래킹
- DeepSORT 알고리즘 사용
- 다중 객체 동시 추적
- 안정적인 ID 유지

### 3. 실시간 시각화
- 바운딩 박스 및 라벨 표시
- 트래킹 궤적 표시
- 실시간 성능 모니터링

### 4. 결과 저장
- 처리된 비디오 저장
- 텍스트 형태의 감지/트래킹 결과
- 스크린샷 및 통계 정보

## 키보드 단축키

실행 중 사용 가능한 키:
- `q`: 종료
- `r`: 트래커 리셋
- `s`: 현재 프레임 스크린샷 저장

## 출력 파일

처리 결과는 `data/outputs/` 폴더에 저장됩니다:
- 처리된 비디오 파일
- 텍스트 형태의 감지 결과
- 스크린샷 이미지

## 성능 최적화

### GPU 사용
NVIDIA GPU가 있는 경우 자동으로 CUDA를 사용하여 성능이 향상됩니다.

### 모델 크기 조정
`utils/config.py`에서 YOLO 모델을 변경할 수 있습니다:
- `yolov8n.pt`: 가장 빠름, 낮은 정확도
- `yolov8s.pt`: 균형
- `yolov8m.pt`: 높은 정확도
- `yolov8l.pt`: 매우 높은 정확도
- `yolov8x.pt`: 최고 정확도, 가장 느림

## 문제 해결

### 일반적인 문제

1. **ImportError: No module named 'torch'**
   - PyTorch 설치: `pip install torch torchvision`

2. **웹캠이 인식되지 않음**
   - 다른 응용프로그램에서 웹캠 사용 중인지 확인
   - 웹캠 드라이버 업데이트

3. **CUDA out of memory**
   - 더 작은 YOLO 모델 사용 (yolov8n)
   - 입력 해상도 줄이기

4. **느린 처리 속도**
   - GPU 사용 확인
   - 불필요한 클래스 필터링
   - 신뢰도 임계값 높이기

### 시스템 요구사항 확인
```bash
python test_system.py
```

## 라이센스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 지원

문제가 발생하면 다음을 확인해주세요:
1. Python 버전 (3.8 이상)
2. 필요한 패키지 설치 여부
3. 웹캠 연결 상태
4. GPU 드라이버 (CUDA 사용 시)
