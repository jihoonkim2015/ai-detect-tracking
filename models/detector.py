"""
객체 감지 모델 클래스
YOLOv8을 사용한 실시간 객체 감지
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from utils.config import Config

class ObjectDetector:
    def __init__(self, model_path=None):
        """
        객체 감지기 초기화
        
        Args:
            model_path (str): YOLO 모델 경로 (None인 경우 기본 모델 사용)
        """
        self.model_path = model_path or Config.YOLO_MODEL
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.IOU_THRESHOLD
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # YOLO 모델 로드
        print(f"객체 감지 모델 로딩 중: {self.model_path}")
        print(f"사용 디바이스: {self.device}")
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print("모델 로딩 완료!")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
        
        # 클래스 이름 매핑
        self.class_names = self.model.names
        
    def detect(self, frame):
        """
        프레임에서 객체 감지 수행
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            
        Returns:
            list: 감지된 객체 정보 리스트 [x1, y1, x2, y2, confidence, class_id]
        """
        try:
            # YOLO 추론 수행
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            
            # 결과 파싱
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 신뢰도 및 클래스 ID
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 결과 리스트에 추가
                        detections.append([
                            int(x1), int(y1), int(x2), int(y2),
                            float(confidence), class_id
                        ])
            
            return detections
            
        except Exception as e:
            print(f"객체 감지 중 오류 발생: {e}")
            return []
    
    def get_class_name(self, class_id):
        """
        클래스 ID로부터 클래스 이름 반환
        
        Args:
            class_id (int): 클래스 ID
            
        Returns:
            str: 클래스 이름
        """
        return self.class_names.get(class_id, f"Class_{class_id}")
    
    def filter_detections(self, detections, target_classes=None):
        """
        특정 클래스만 필터링
        
        Args:
            detections (list): 감지 결과 리스트
            target_classes (list): 필터링할 클래스 이름 리스트
            
        Returns:
            list: 필터링된 감지 결과
        """
        if target_classes is None:
            return detections
        
        filtered = []
        for detection in detections:
            class_id = detection[5]
            class_name = self.get_class_name(class_id)
            
            if class_name in target_classes:
                filtered.append(detection)
        
        return filtered
    
    def draw_detections(self, frame, detections, show_confidence=True):
        """
        프레임에 감지 결과 그리기
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            detections (list): 감지 결과 리스트
            show_confidence (bool): 신뢰도 표시 여부
            
        Returns:
            numpy.ndarray: 감지 결과가 그려진 프레임
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # 클래스 이름 및 색상
            class_name = self.get_class_name(class_id)
            color = Config.COLORS[class_id % len(Config.COLORS)]
            
            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, Config.BOX_THICKNESS)
            
            # 라벨 텍스트 준비
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # 라벨 배경 그리기
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       Config.FONT_SCALE, Config.FONT_THICKNESS)[0]
            
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # 라벨 텍스트 그리기
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       Config.FONT_SCALE,
                       (255, 255, 255),
                       Config.FONT_THICKNESS)
        
        return annotated_frame
