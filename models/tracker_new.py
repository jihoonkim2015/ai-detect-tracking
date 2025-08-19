"""
객체 트래킹 모델 클래스 (간소화 버전)
DeepSORT 대신 간단하고 안정적인 트래커 사용
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
        
        # 간단한 트래커 사용
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
        """
        if not detections:
            self._age_tracks()
            return self._get_active_tracks()
        
        matched_tracks = []
        used_detections = set()
        
        # 기존 트랙과 감지 결과 매칭
        for track_id, track_info in list(self.tracks.items()):
            if track_info['disappeared'] > 0:
                continue
                
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            track_center = self._get_center(track_info['bbox'])
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                det_center = self._get_center(detection[:4])
                distance = self._calculate_distance(track_center, det_center)
                
                if distance < best_distance and distance < 100:
                    best_distance = distance
                    best_match = detection
                    best_idx = i
            
            if best_match is not None:
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
        
        # 매칭되지 않은 트랙 나이 증가
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
        
        self._update_track_history(matched_tracks)
        return matched_tracks
    
    def _get_center(self, bbox):
        """바운딩 박스의 중심점 계산"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, center1, center2):
        """두 중심점 사이의 거리 계산"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _age_tracks(self):
        """모든 트랙의 나이 증가"""
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['disappeared'] += 1
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                if track_id in self.active_tracks:
                    self.active_tracks.remove(track_id)
                del self.tracks[track_id]
    
    def _get_active_tracks(self):
        """활성 트랙들 반환"""
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
        """트래킹 히스토리 업데이트"""
        for track in tracks:
            track_id = track[4]
            x1, y1, x2, y2 = track[:4]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((center_x, center_y))
            
            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id] = self.track_history[track_id][-50:]
    
    def draw_tracks(self, frame, tracks, show_trajectory=True, show_id=True):
        """프레임에 트래킹 결과 그리기"""
        annotated_frame = frame.copy()
        
        if show_trajectory:
            self._draw_trajectories(annotated_frame)
        
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, confidence = track
            
            color = Config.COLORS[track_id % len(Config.COLORS)]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, Config.BOX_THICKNESS + 1)
            
            if show_id:
                label = f"ID: {track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           Config.FONT_SCALE, Config.FONT_THICKNESS)[0]
                
                cv2.rectangle(annotated_frame,
                             (x1, y2),
                             (x1 + label_size[0], y2 + label_size[1] + 10),
                             color, -1)
                
                cv2.putText(annotated_frame, label,
                           (x1, y2 + label_size[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           Config.FONT_SCALE,
                           (255, 255, 255),
                           Config.FONT_THICKNESS)
        
        return annotated_frame
    
    def _draw_trajectories(self, frame):
        """객체 궤적 그리기"""
        for track_id, history in self.track_history.items():
            if track_id not in self.active_tracks:
                continue
            
            if len(history) < 2:
                continue
            
            color = Config.COLORS[track_id % len(Config.COLORS)]
            
            for i in range(1, len(history)):
                cv2.line(frame, history[i-1], history[i], color, 2)
            
            if history:
                cv2.circle(frame, history[-1], 3, color, -1)
    
    def get_track_count(self):
        """현재 활성 트랙 수 반환"""
        return len(self.active_tracks)
    
    def reset(self):
        """트래커 리셋"""
        self.tracks = {}
        self.next_id = 1
        self.track_history = {}
        self.active_tracks = set()
        print("트래커가 리셋되었습니다.")
