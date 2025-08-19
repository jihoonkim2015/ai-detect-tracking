"""
AI 객체 감지 및 트래킹 시스템 설정
"""

class Config:
    # YOLOv8 모델 설정
    YOLO_MODEL = "yolov8n.pt"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # DeepSORT 트래킹 설정
    MAX_AGE = 70  # 객체가 사라진 후 추적을 유지할 최대 프레임 수
    MIN_HITS = 3  # 트래킹을 시작하기 위한 최소 감지 횟수
    IOU_THRESHOLD_TRACKER = 0.3
    
    # 비디오 설정
    VIDEO_FPS = 30
    VIDEO_CODEC = 'mp4v'
    
    # 출력 설정
    OUTPUT_DIR = "data/outputs"
    SAVE_VIDEO = True
    SAVE_FRAMES = False
    SHOW_DISPLAY = True
    
    # 감지할 클래스 (COCO 데이터셋 기준)
    DETECT_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 색상 설정 (BGR 형식)
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128)
    ]
    
    # UI 설정
    FONT = "cv2.FONT_HERSHEY_SIMPLEX"
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2
    
    # 성능 모니터링
    SHOW_FPS = True
    SHOW_COUNT = True
