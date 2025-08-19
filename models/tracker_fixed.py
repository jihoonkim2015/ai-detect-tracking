"""
객체 트래킹 모델 클래스 (간소화 버전)
DeepSORT 문제 발생시 간단한 트래커 사용
"""

import cv2
import numpy as np
from utils.config import Config

class ObjectTracker:
    def __init__(self):
        """
        객체 트래커 초기화
        """
        print("객체 트래커 초기화 중...")
        
        # 간단한 트래커 사용 (DeepSORT 대안)
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 30
        
        # 트래킹 통계
        self.track_history = {}
        self.active_tracks = set()
        
        print("간단한 객체 트래커 초기화 완료!")
    
    def update(self, detections, frame=None):
        """
        감지 결과를 바탕으로 트래킹 업데이트
        
        Args:
            detections (list): 감지 결과 리스트 [x1, y1, x2, y2, confidence, class_id]
            frame (numpy.ndarray): 현재 프레임 (사용하지 않음)
            
        Returns:
            list: 트래킹 결과 리스트 [x1, y1, x2, y2, track_id, class_id, confidence]
        """
        if not detections:
            # 감지된 객체가 없으면 기존 트랙들의 나이 증가
            self._age_tracks()
            return self._get_active_tracks()
        
        # 현재 트랙들과 새 감지 결과 매칭
        matched_tracks = []
        used_detections = set()
        
        # 기존 트랙과 감지 결과 매칭
        for track_id, track_info in list(self.tracks.items()):
            if track_info['disappeared'] > 0:
                continue  # 이미 사라진 트랙은 스킵
                
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            track_center = self._get_center(track_info['bbox'])
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                det_center = self._get_center(detection[:4])
                distance = self._calculate_distance(track_center, det_center)
                
                # IoU 기반 매칭도 고려
                iou = self._calculate_iou(track_info['bbox'], detection[:4])
                
                # 거리와 IoU를 결합한 점수
                score = distance - (iou * 100)  # IoU가 높을수록 점수 낮음
                
                if score < best_distance and distance < 100:  # 100픽셀 이내
                    best_distance = score
                    best_match = detection
                    best_idx = i
            
            if best_match is not None:
                # 트랙 업데이트
                x1, y1, x2, y2, confidence, class_id = best_match
                self.tracks[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'confidence': confidence,
                    'disappeared': 0
                }
                
                matched_tracks.append([x1, y1, x2, y2, track_id, class_id, confidence])
                used_detections.add(best_idx)
                self.active_tracks.add(track_id)
        
        # 매칭되지 않은 기존 트랙들의 나이 증가
        for track_id, track_info in self.tracks.items():
            if track_id not in [t[4] for t in matched_tracks]:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    if track_id in self.active_tracks:
                        self.active_tracks.remove(track_id)
        
        # 새로운 감지 결과를 새 트랙으로 추가
        for i, detection in enumerate(detections):
            if i not in used_detections:
                x1, y1, x2, y2, confidence, class_id = detection
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'confidence': confidence,
                    'disappeared': 0
                }
                
                matched_tracks.append([x1, y1, x2, y2, track_id, class_id, confidence])
                self.active_tracks.add(track_id)
        
        # 트래킹 히스토리 업데이트
        self._update_track_history(matched_tracks)
        
        return matched_tracks
    
    def _get_center(self, bbox):
        """
        바운딩 박스의 중심점 계산
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, center1, center2):
        """
        두 중심점 사이의 유클리드 거리 계산
        """
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        두 바운딩 박스의 IoU 계산
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역 계산
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 각 박스의 면적
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 합집합 면적
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _age_tracks(self):
        """
        모든 트랙의 나이 증가
        """
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['disappeared'] += 1
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                if track_id in self.active_tracks:
                    self.active_tracks.remove(track_id)
                del self.tracks[track_id]
    
    def _get_active_tracks(self):
        """
        활성 트랙들 반환
        """
        active_tracks = []
        for track_id, track_info in self.tracks.items():
            if track_info['disappeared'] == 0:
                bbox = track_info['bbox']
                active_tracks.append([
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    track_id, track_info['class_id'], track_info['confidence']
                ])
        return active_tracks
    
    def _update_track_history(self, tracks):
        """
        트래킹 히스토리 업데이트
        """
        for track in tracks:
            track_id = track[4]
            x1, y1, x2, y2 = track[:4]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((center_x, center_y))
            
            # 히스토리 길이 제한 (메모리 절약)
            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id] = self.track_history[track_id][-50:]
    
    def draw_tracks(self, frame, tracks, show_trajectory=True, show_id=True):
        """
        프레임에 트래킹 결과 그리기
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            tracks (list): 트래킹 결과 리스트
            show_trajectory (bool): 궤적 표시 여부
            show_id (bool): 트랙 ID 표시 여부
            
        Returns:
            numpy.ndarray: 트래킹 결과가 그려진 프레임
        """
        annotated_frame = frame.copy()
        
        # 트래킹 궤적 그리기
        if show_trajectory:
            self._draw_trajectories(annotated_frame)
        
        # 트래킹 박스 및 ID 그리기
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, confidence = track
            
            # 색상 선택 (트랙 ID 기반)
            color = Config.COLORS[track_id % len(Config.COLORS)]
            
            # 바운딩 박스 그리기 (두꺼운 선으로)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, Config.BOX_THICKNESS + 1)
            
            if show_id:
                # 트랙 ID 표시
                label = f"ID: {track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           Config.FONT_SCALE, Config.FONT_THICKNESS)[0]
                
                # ID 라벨 배경
                cv2.rectangle(annotated_frame,
                             (x1, y2),
                             (x1 + label_size[0], y2 + label_size[1] + 10),
                             color, -1)
                
                # ID 텍스트
                cv2.putText(annotated_frame, label,
                           (x1, y2 + label_size[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           Config.FONT_SCALE,
                           (255, 255, 255),
                           Config.FONT_THICKNESS)
        
        return annotated_frame
    
    def _draw_trajectories(self, frame):
        """
        객체 궤적 그리기
        """
        for track_id, history in self.track_history.items():
            if track_id not in self.active_tracks:
                continue
            
            if len(history) < 2:
                continue
            
            # 궤적 색상
            color = Config.COLORS[track_id % len(Config.COLORS)]
            
            # 궤적 선 그리기
            for i in range(1, len(history)):
                cv2.line(frame, history[i-1], history[i], color, 2)
            
            # 현재 위치에 점 그리기
            if history:
                cv2.circle(frame, history[-1], 3, color, -1)
    
    def get_track_count(self):
        """
        현재 활성 트랙 수 반환
        """
        return len(self.active_tracks)
    
    def reset(self):
        """
        트래커 리셋
        """
        self.tracks = {}
        self.next_id = 1
        self.track_history = {}
        self.active_tracks = set()
        print("트래커가 리셋되었습니다.")
