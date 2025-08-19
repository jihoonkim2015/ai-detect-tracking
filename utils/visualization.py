"""
시각화 관련 유틸리티 함수들
"""

import cv2
import numpy as np
from utils.config import Config

class Visualizer:
    def __init__(self):
        """
        시각화 도구 초기화
        """
        self.colors = Config.COLORS
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = Config.FONT_SCALE
        self.font_thickness = Config.FONT_THICKNESS
        self.box_thickness = Config.BOX_THICKNESS
    
    def draw_info_panel(self, frame, fps=0, frame_count=0, track_count=0, 
                       detection_count=0, elapsed_time=0):
        """
        정보 패널 그리기
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            fps (float): FPS
            frame_count (int): 프레임 번호
            track_count (int): 트래킹 수
            detection_count (int): 감지 수
            elapsed_time (float): 경과 시간
            
        Returns:
            numpy.ndarray: 정보가 그려진 프레임
        """
        height, width = frame.shape[:2]
        panel_height = 150
        panel_width = 300
        
        # 반투명 패널 생성
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), 
                     (0, 0, 0), -1)
        
        # 투명도 적용
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 정보 텍스트
        y_offset = 35
        line_height = 25
        
        info_texts = [
            f"FPS: {fps:.1f}",
            f"Frame: {frame_count}",
            f"Detections: {detection_count}",
            f"Tracks: {track_count}",
            f"Time: {elapsed_time:.1f}s"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = y_offset + i * line_height
            cv2.putText(frame, text, (20, y_pos), self.font, 
                       self.font_scale, (255, 255, 255), self.font_thickness)
        
        return frame
    
    def draw_detection_stats(self, frame, class_counts):
        """
        클래스별 감지 통계 그리기
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            class_counts (dict): 클래스별 감지 수
            
        Returns:
            numpy.ndarray: 통계가 그려진 프레임
        """
        if not class_counts:
            return frame
        
        height, width = frame.shape[:2]
        
        # 통계 패널 크기 계산
        panel_width = 250
        panel_height = min(300, 30 + len(class_counts) * 25)
        
        # 패널 위치 (우측 상단)
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # 반투명 패널 생성
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        
        # 투명도 적용
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 제목
        cv2.putText(frame, "Detection Stats", (panel_x + 10, panel_y + 25), 
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # 클래스별 통계
        y_offset = panel_y + 50
        for i, (class_name, count) in enumerate(class_counts.items()):
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (panel_x + 10, y_offset + i * 25), 
                       self.font, self.font_scale - 0.1, (255, 255, 255), 
                       self.font_thickness)
        
        return frame
    
    def draw_progress_bar(self, frame, progress, total_frames=0):
        """
        진행률 바 그리기
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            progress (float): 진행률 (0.0 ~ 1.0)
            total_frames (int): 총 프레임 수
            
        Returns:
            numpy.ndarray: 진행률 바가 그려진 프레임
        """
        if progress <= 0:
            return frame
        
        height, width = frame.shape[:2]
        
        # 진행률 바 설정
        bar_width = width - 40
        bar_height = 20
        bar_x = 20
        bar_y = height - 40
        
        # 배경 바
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # 진행률 바
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # 진행률 텍스트
        if total_frames > 0:
            text = f"Progress: {progress*100:.1f}% ({int(progress*total_frames)}/{total_frames})"
        else:
            text = f"Progress: {progress*100:.1f}%"
        
        cv2.putText(frame, text, (bar_x, bar_y - 5), self.font, 
                   self.font_scale, (255, 255, 255), self.font_thickness)
        
        return frame
    
    def draw_crosshair(self, frame, center=None, size=20, color=(0, 255, 0)):
        """
        십자선 그리기
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            center (tuple): 중심점 좌표 (None인 경우 화면 중앙)
            size (int): 십자선 크기
            color (tuple): 색상
            
        Returns:
            numpy.ndarray: 십자선이 그려진 프레임
        """
        height, width = frame.shape[:2]
        
        if center is None:
            center = (width // 2, height // 2)
        
        cx, cy = center
        
        # 수평선
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
        # 수직선
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)
        
        return frame
    
    def draw_zone(self, frame, zone_points, zone_name="Zone", color=(255, 255, 0)):
        """
        관심 영역 그리기
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            zone_points (list): 영역 좌표점 리스트
            zone_name (str): 영역 이름
            color (tuple): 색상
            
        Returns:
            numpy.ndarray: 영역이 그려진 프레임
        """
        if len(zone_points) < 3:
            return frame
        
        # 폴리곤 그리기
        points = np.array(zone_points, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # 반투명 영역
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # 경계선
        cv2.polylines(frame, [points], True, color, 3)
        
        # 영역 이름
        if zone_points:
            text_x = min(zone_points, key=lambda p: p[0])[0]
            text_y = min(zone_points, key=lambda p: p[1])[1] - 10
            cv2.putText(frame, zone_name, (text_x, text_y), self.font,
                       self.font_scale, color, self.font_thickness)
        
        return frame
    
    def create_heatmap(self, detections_history, frame_shape, alpha=0.6):
        """
        감지 히트맵 생성
        
        Args:
            detections_history (list): 감지 이력 리스트
            frame_shape (tuple): 프레임 크기 (height, width)
            alpha (float): 투명도
            
        Returns:
            numpy.ndarray: 히트맵 이미지
        """
        height, width = frame_shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # 감지 이력을 바탕으로 히트맵 생성
        for detection in detections_history:
            x1, y1, x2, y2 = detection[:4]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 가우시안 분포로 히트 추가
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
            heatmap[mask] += 1
        
        # 정규화
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 컬러맵 적용
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        
        return heatmap_colored
    
    def add_watermark(self, frame, text="AI Detection & Tracking", 
                     position='bottom-right', alpha=0.7):
        """
        워터마크 추가
        
        Args:
            frame (numpy.ndarray): 입력 프레임
            text (str): 워터마크 텍스트
            position (str): 위치 ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            alpha (float): 투명도
            
        Returns:
            numpy.ndarray: 워터마크가 추가된 프레임
        """
        height, width = frame.shape[:2]
        
        # 텍스트 크기 계산
        text_size = cv2.getTextSize(text, self.font, self.font_scale, 
                                   self.font_thickness)[0]
        
        # 위치 계산
        margin = 10
        if position == 'top-left':
            x, y = margin, margin + text_size[1]
        elif position == 'top-right':
            x, y = width - text_size[0] - margin, margin + text_size[1]
        elif position == 'bottom-left':
            x, y = margin, height - margin
        else:  # bottom-right
            x, y = width - text_size[0] - margin, height - margin
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - text_size[1] - 5), 
                     (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 텍스트 추가
        cv2.putText(frame, text, (x, y), self.font, self.font_scale, 
                   (255, 255, 255), self.font_thickness)
        
        return frame
