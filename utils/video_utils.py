"""
비디오 처리 관련 유틸리티 함수들
"""

import cv2
import os
import time
from pathlib import Path
from utils.config import Config

class VideoProcessor:
    def __init__(self, source=0, output_path=None, desired_width=None, desired_height=None):
        """
        비디오 프로세서 초기화
        
        Args:
            source (str/int): 비디오 소스 (0: 웹캠, 문자열: 파일 경로)
            output_path (str): 출력 파일 경로 (None인 경우 자동 생성)
            desired_width (int): 캡처에 설정할 너비 (웹캠 전용, 선택)
            desired_height (int): 캡처에 설정할 높이 (웹캠 전용, 선택)
        """
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.fps = Config.VIDEO_FPS
        self.frame_count = 0
        self.total_frames = 0
        self.desired_width = desired_width
        self.desired_height = desired_height
        
        # 입력 소스 초기화
        self._initialize_source()
        
        # 출력 설정
        if Config.SAVE_VIDEO and self.output_path:
            self._initialize_writer()
    
    def _initialize_source(self):
        """
        비디오 입력 소스 초기화
        """
        print(f"비디오 소스 초기화: {self.source}")
        
        self.cap = cv2.VideoCapture(self.source)
        
        # 원하는 해상도가 지정된 경우 캡처 속성으로 설정 (웹캠에만 적용됨)
        if self.cap.isOpened() and (self.desired_width or self.desired_height):
            if self.desired_width:
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.desired_width))
                except Exception:
                    pass
            if self.desired_height:
                try:
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.desired_height))
                except Exception:
                    pass
        
        if not self.cap.isOpened():
            raise ValueError(f"비디오 소스를 열 수 없습니다: {self.source}")
        
        # 비디오 정보 가져오기
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or Config.VIDEO_FPS
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"비디오 정보 - FPS: {self.fps}, 해상도: {self.width}x{self.height}")
        if self.total_frames > 0:
            print(f"총 프레임 수: {self.total_frames}")
    
    def _initialize_writer(self):
        """
        비디오 출력 라이터 초기화
        """
        if not self.output_path:
            return
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 비디오 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
        
        # 비디오 라이터 초기화
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        
        if self.writer.isOpened():
            print(f"출력 비디오 설정 완료: {self.output_path}")
        else:
            print(f"출력 비디오 설정 실패: {self.output_path}")
            self.writer = None
    
    def read_frame(self):
        """
        다음 프레임 읽기
        
        Returns:
            tuple: (성공 여부, 프레임)
        """
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame
    
    def write_frame(self, frame):
        """
        프레임을 출력 비디오에 쓰기
        
        Args:
            frame (numpy.ndarray): 출력할 프레임
        """
        if self.writer is not None:
            self.writer.write(frame)
    
    def get_progress(self):
        """
        처리 진행률 반환 (비디오 파일인 경우)
        
        Returns:
            float: 진행률 (0.0 ~ 1.0)
        """
        if self.total_frames > 0:
            return min(self.frame_count / self.total_frames, 1.0)
        return 0.0
    
    def get_frame_info(self):
        """
        현재 프레임 정보 반환
        
        Returns:
            dict: 프레임 정보
        """
        return {
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'progress': self.get_progress()
        }
    
    def release(self):
        """
        리소스 해제
        """
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

class FPSCounter:
    def __init__(self, window_size=30):
        """
        FPS 카운터 초기화
        
        Args:
            window_size (int): FPS 계산을 위한 윈도우 크기
        """
        self.window_size = window_size
        self.frame_times = []
        self.start_time = time.time()
    
    def update(self):
        """
        FPS 업데이트
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # 윈도우 크기 유지
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """
        현재 FPS 반환
        
        Returns:
            float: FPS 값
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_diff
    
    def get_elapsed_time(self):
        """
        경과 시간 반환
        
        Returns:
            float: 시작부터 경과된 시간 (초)
        """
        return time.time() - self.start_time

def create_output_path(source, suffix="processed"):
    """
    입력 소스에 기반한 출력 경로 생성
    
    Args:
        source (str/int): 입력 소스
        suffix (str): 파일명 접미사
        
    Returns:
        str: 출력 파일 경로
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if isinstance(source, int) or source == '0':
        # 웹캠인 경우
        filename = f"webcam_{suffix}_{timestamp}.mp4"
    else:
        # 파일인 경우
        source_path = Path(source)
        filename = f"{source_path.stem}_{suffix}_{timestamp}{source_path.suffix}"
    
    return os.path.join(Config.OUTPUT_DIR, filename)

def is_image_file(filepath):
    """
    이미지 파일 여부 확인
    
    Args:
        filepath (str): 파일 경로
        
    Returns:
        bool: 이미지 파일 여부
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    return Path(filepath).suffix.lower() in image_extensions

def is_video_file(filepath):
    """
    비디오 파일 여부 확인
    
    Args:
        filepath (str): 파일 경로
        
    Returns:
        bool: 비디오 파일 여부
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(filepath).suffix.lower() in video_extensions

def resize_frame(frame, max_width=1280, max_height=720):
    """
    프레임 크기 조정 (비율 유지)
    
    Args:
        frame (numpy.ndarray): 입력 프레임
        max_width (int): 최대 너비
        max_height (int): 최대 높이
        
    Returns:
        numpy.ndarray: 크기 조정된 프레임
    """
    height, width = frame.shape[:2]
    
    # 크기 조정이 필요한지 확인
    if width <= max_width and height <= max_height:
        return frame
    
    # 비율 계산
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    # 새로운 크기 계산
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 크기 조정
    resized_frame = cv2.resize(frame, (new_width, new_height), 
                              interpolation=cv2.INTER_AREA)
    
    return resized_frame
