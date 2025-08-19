"""
객체 트래킹 모델 클래스
DeepSORT 알고리즘을 사용한 다중 객체 추적
"""

import cv2
import numpy as np

# DeepSORT import 시도 (여러 방법)
DeepSort = None
DEEPSORT_AVAILABLE = False

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
    print("DeepSort import 성공: deep_sort_realtime.DeepSort")
except ImportError:
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        DEEPSORT_AVAILABLE = True
        print("DeepSort import 성공: deep_sort_realtime.deepsort_tracker.DeepSort")
    except ImportError:
        try:
            from deep_sort_realtime.deep_sort.deep_sort import DeepSort
            DEEPSORT_AVAILABLE = True
            print("DeepSort import 성공: deep_sort_realtime.deep_sort.deep_sort.DeepSort")
        except ImportError:
            try:
                # Alternative: norfair tracker
                import norfair
                DEEPSORT_AVAILABLE = False
                print("DeepSort를 사용할 수 없습니다. Norfair 트래커를 시도합니다.")
            except ImportError:
                DEEPSORT_AVAILABLE = False
                print("외부 트래킹 라이브러리를 사용할 수 없습니다. 내장 간단한 트래커를 사용합니다.")

from utils.config import Config

class ObjectTracker:
    def __init__(self):
        """
        객체 트래커 초기화
        """
        print("객체 트래커 초기화 중...")
        
        if not DEEPSORT_AVAILABLE or DeepSort is None:
            print("Warning: DeepSORT를 사용할 수 없습니다. 간단한 트래커를 사용합니다.")
            self.use_simple_tracker = True
            self.tracker = None
            self.simple_tracks = {}
            self.next_id = 1
        else:
            self.use_simple_tracker = False
            # DeepSORT 트래커 초기화
            try:
                self.tracker = DeepSort(
                    max_age=Config.MAX_AGE,
                    n_init=Config.MIN_HITS,
                    nms_max_overlap=Config.IOU_THRESHOLD_TRACKER,
                    max_cosine_distance=0.2,
                    nn_budget=None,
                    override_track_class=None,
                    embedder="mobilenet",
                    half=True,
                    bgr=True,
                    embedder_gpu=True,
                    embedder_model_name=None,
                    polygons=False,
                    today=None
                )
                print("DeepSORT 트래커 초기화 완료!")
            except Exception as e:
                print(f"DeepSORT 초기화 실패: {e}")
                print("간단한 트래커를 사용합니다.")
                self.use_simple_tracker = True
                self.tracker = None
                self.simple_tracks = {}
                self.next_id = 1
        
        # 트래킹 통계
        self.track_history = {}
        self.active_tracks = set()
    
    def update(self, detections, frame):
        """
        감지 결과를 바탕으로 트래킹 업데이트
        
        Args:
            detections (list): 감지 결과 리스트 [x1, y1, x2, y2, confidence, class_id]
            frame (numpy.ndarray): 현재 프레임
            
        Returns:
            list: 트래킹 결과 리스트 [x1, y1, x2, y2, track_id, class_id, confidence]
        """
        if self.use_simple_tracker:
            return self._simple_tracker_update(detections)
        
        if not detections:
            # 감지된 객체가 없어도 기존 트랙 업데이트
            tracks = self.tracker.update_tracks([], frame=frame)
            return self._format_tracks(tracks)
        
        # DeepSORT 형식으로 변환
        bbs_ids = []
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # (left, top, width, height) 형식으로 변환
            left = x1
            top = y1
            width = x2 - x1
            height = y2 - y1
            
            bbs_ids.append(([left, top, width, height], confidence, class_id))
        
        # 트래킹 업데이트
        tracks = self.tracker.update_tracks(bbs_ids, frame=frame)
        
        # 트래킹 히스토리 업데이트
        self._update_track_history(tracks)
        
        return self._format_tracks(tracks)
    
    def _simple_tracker_update(self, detections):
        """
        간단한 트래커 업데이트 (DeepSORT 대안)
        
        Args:
            detections (list): 감지 결과 리스트
            
        Returns:
            list: 트래킹 결과 리스트
        """
        if not detections:
            # 감지된 객체가 없으면 기존 트랙들을 페이드아웃
            expired_tracks = []
            for track_id in list(self.simple_tracks.keys()):
                self.simple_tracks[track_id]['age'] += 1
                if self.simple_tracks[track_id]['age'] > 30:  # 30프레임 후 삭제
                    expired_tracks.append(track_id)
            
            for track_id in expired_tracks:
                del self.simple_tracks[track_id]
                if track_id in self.active_tracks:
                    self.active_tracks.remove(track_id)
            
            return []
        
        tracks = []
        used_detections = set()
        
        # 기존 트랙과 새 감지 결과 매칭
        for track_id, track_info in list(self.simple_tracks.items()):
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            track_center = track_info['center']
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                x1, y1, x2, y2 = detection[:4]
                det_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # 유클리드 거리 계산
                distance = np.sqrt((track_center[0] - det_center[0])**2 + 
                                 (track_center[1] - det_center[1])**2)
                
                if distance < best_distance and distance < 100:  # 100픽셀 이내
                    best_distance = distance
                    best_match = detection
                    best_idx = i
            
            if best_match is not None:
                # 트랙 업데이트
                x1, y1, x2, y2, confidence, class_id = best_match
                self.simple_tracks[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'class_id': class_id,
                    'confidence': confidence,
                    'age': 0
                }
                
                tracks.append([x1, y1, x2, y2, track_id, class_id, confidence])
                used_detections.add(best_idx)
            else:
                # 매칭되지 않은 트랙의 나이 증가
                self.simple_tracks[track_id]['age'] += 1
                if self.simple_tracks[track_id]['age'] > 10:
                    # 10프레임 후 삭제
                    if track_id in self.active_tracks:
                        self.active_tracks.remove(track_id)
                    continue
        
        # 새로운 감지 결과를 새 트랙으로 추가
        for i, detection in enumerate(detections):
            if i not in used_detections:
                x1, y1, x2, y2, confidence, class_id = detection
                track_id = self.next_id
                self.next_id += 1
                
                self.simple_tracks[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'class_id': class_id,
                    'confidence': confidence,
                    'age': 0
                }
                
                tracks.append([x1, y1, x2, y2, track_id, class_id, confidence])
                self.active_tracks.add(track_id)
        
        # 히스토리 업데이트 (간단한 버전)
        for track in tracks:
            track_id = track[4]
            x1, y1, x2, y2 = track[:4]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((center_x, center_y))
            
            # 히스토리 길이 제한
            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id] = self.track_history[track_id][-50:]
        
        return tracks
    
    def _format_tracks(self, tracks):
        """
        트래킹 결과를 표준 형식으로 변환
        
        Args:
            tracks (list): DeepSORT 트래킹 결과
            
        Returns:
            list: 표준화된 트래킹 결과
        """
        formatted_tracks = []
        current_track_ids = set()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            # 바운딩 박스 정보 추출
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            track_id = track.track_id
            current_track_ids.add(track_id)
            
            # 클래스 정보 (감지 시점의 클래스 사용)
            class_id = getattr(track, 'det_class', 0)
            confidence = getattr(track, 'det_conf', 0.0)
            
            formatted_tracks.append([x1, y1, x2, y2, track_id, class_id, confidence])
        
        # 활성 트랙 ID 업데이트
        self.active_tracks = current_track_ids
        
        return formatted_tracks
    
    def _update_track_history(self, tracks):
        """
        트래킹 히스토리 업데이트
        
        Args:
            tracks (list): 현재 트래킹 결과
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            center_x = int((ltrb[0] + ltrb[2]) / 2)
            center_y = int((ltrb[1] + ltrb[3]) / 2)
            
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
        
        Args:
            frame (numpy.ndarray): 입력 프레임
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
        
        Returns:
            int: 활성 트랙 수
        """
        return len(self.active_tracks)
    
    def reset(self):
        """
        트래커 리셋
        """
        if self.use_simple_tracker:
            self.simple_tracks = {}
            self.next_id = 1
        else:
            self.tracker = DeepSort(
                max_age=Config.MAX_AGE,
                n_init=Config.MIN_HITS,
                nms_max_overlap=Config.IOU_THRESHOLD_TRACKER,
                max_cosine_distance=0.2,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                polygons=False,
                today=None
            )
        self.track_history = {}
        self.active_tracks = set()
