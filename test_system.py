"""
시스템 테스트 스크립트
모든 컴포넌트가 정상적으로 작동하는지 확인
"""

import sys
import importlib
import cv2
import numpy as np
import torch

def test_imports():
    """
    필요한 라이브러리 import 테스트
    """
    print("1. 라이브러리 import 테스트...")
    
    required_modules = [
        'torch', 'torchvision', 'cv2', 'numpy', 
        'ultralytics', 'deep_sort_realtime'
    ]
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   ✓ {module}")
        except ImportError as e:
            print(f"   ✗ {module} - 오류: {e}")
            return False
    
    return True

def test_torch():
    """
    PyTorch 설치 및 CUDA 지원 테스트
    """
    print("\n2. PyTorch 테스트...")
    
    print(f"   PyTorch 버전: {torch.__version__}")
    print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   GPU 개수: {torch.cuda.device_count()}")
        print(f"   현재 GPU: {torch.cuda.get_device_name()}")
    
    # 간단한 텐서 연산 테스트
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("   ✓ 기본 텐서 연산 성공")
        
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print("   ✓ GPU 텐서 연산 성공")
    
    except Exception as e:
        print(f"   ✗ 텐서 연산 실패: {e}")
        return False
    
    return True

def test_opencv():
    """
    OpenCV 테스트
    """
    print("\n3. OpenCV 테스트...")
    
    print(f"   OpenCV 버전: {cv2.__version__}")
    
    # 기본 이미지 처리 테스트
    try:
        # 더미 이미지 생성
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (255, 0, 0)  # 파란색
        
        # 기본 연산
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, (50, 50))
        
        print("   ✓ 기본 이미지 처리 성공")
        
    except Exception as e:
        print(f"   ✗ 이미지 처리 실패: {e}")
        return False
    
    return True

def test_yolo():
    """
    YOLO 모델 로딩 테스트
    """
    print("\n4. YOLO 모델 테스트...")
    
    try:
        from ultralytics import YOLO
        
        # YOLOv8n 모델 로드 (가장 가벼운 모델)
        model = YOLO('yolov8n.pt')
        print("   ✓ YOLO 모델 로딩 성공")
        
        # 더미 이미지로 추론 테스트
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        print("   ✓ YOLO 추론 성공")
        
    except Exception as e:
        print(f"   ✗ YOLO 테스트 실패: {e}")
        return False
    
    return True

def test_deepsort():
    """
    DeepSORT 테스트
    """
    print("\n5. DeepSORT 테스트...")
    
    try:
        from deep_sort_realtime import DeepSort
        
        # DeepSORT 초기화
        tracker = DeepSort()
        print("   ✓ DeepSORT 초기화 성공")
        
        # 더미 감지 결과로 업데이트 테스트
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_detections = [([100, 100, 50, 50], 0.9, 0)]  # (bbox, conf, class)
        
        tracks = tracker.update_tracks(dummy_detections, frame=dummy_frame)
        print("   ✓ DeepSORT 업데이트 성공")
        
    except Exception as e:
        print(f"   ✗ DeepSORT 테스트 실패: {e}")
        return False
    
    return True

def test_project_modules():
    """
    프로젝트 모듈 테스트
    """
    print("\n6. 프로젝트 모듈 테스트...")
    
    try:
        # 설정 모듈
        from utils.config import Config
        print("   ✓ 설정 모듈 import 성공")
        
        # 감지기 모듈
        from models.detector import ObjectDetector
        detector = ObjectDetector()
        print("   ✓ 객체 감지기 초기화 성공")
        
        # 트래커 모듈
        from models.tracker import ObjectTracker
        tracker = ObjectTracker()
        print("   ✓ 객체 트래커 초기화 성공")
        
        # 유틸리티 모듈
        from utils.video_utils import VideoProcessor, FPSCounter
        from utils.visualization import Visualizer
        print("   ✓ 유틸리티 모듈 import 성공")
        
        # 간단한 기능 테스트
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)
        tracks = tracker.update(detections, dummy_frame)
        
        visualizer = Visualizer()
        result_frame = visualizer.draw_info_panel(dummy_frame)
        
        print("   ✓ 기본 기능 테스트 성공")
        
    except Exception as e:
        print(f"   ✗ 프로젝트 모듈 테스트 실패: {e}")
        return False
    
    return True

def test_webcam():
    """
    웹캠 접근 테스트
    """
    print("\n7. 웹캠 테스트...")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ⚠ 웹캠을 열 수 없습니다 (연결되지 않았거나 사용 중)")
            return True  # 웹캠이 없어도 시스템은 동작 가능
        
        ret, frame = cap.read()
        if ret:
            print(f"   ✓ 웹캠 접근 성공 (해상도: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print("   ⚠ 웹캠에서 프레임을 읽을 수 없습니다")
        
        cap.release()
        
    except Exception as e:
        print(f"   ⚠ 웹캠 테스트 중 오류: {e}")
    
    return True

def main():
    """
    전체 테스트 실행
    """
    print("=" * 60)
    print("AI 객체 감지 및 트래킹 시스템 테스트")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_torch,
        test_opencv,
        test_yolo,
        test_deepsort,
        test_project_modules,
        test_webcam
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ✗ 테스트 중 예외 발생: {e}")
    
    print("\n" + "=" * 60)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("✅ 모든 테스트가 통과했습니다! 시스템이 정상적으로 작동할 준비가 되었습니다.")
    else:
        print("❌ 일부 테스트가 실패했습니다. 위의 오류 메시지를 확인하고 문제를 해결해주세요.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
