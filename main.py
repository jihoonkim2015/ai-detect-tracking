"""
AI 영상 기반 객체인식 및 트래킹 시스템
PyTorch 기반 YOLOv8 + DeepSORT 구현

사용법:
    python main.py --source 0                    # 웹캠
    python main.py --source video.mp4           # 비디오 파일
    python main.py --source image.jpg           # 이미지 파일
    python main.py --source 0 --save-txt        # 결과 저장
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

# 프로젝트 모듈 import
from models.detector import ObjectDetector
from models.tracker import ObjectTracker
from utils.video_utils import VideoProcessor, FPSCounter, create_output_path, is_image_file, is_video_file
from utils.visualization import Visualizer
from utils.config import Config

class AIDetectionTracker:
    def __init__(self, source=0, output_dir=None, save_txt=False, save_conf=False, cam_width=None, cam_height=None):
        """
        AI 감지 및 트래킹 시스템 초기화
        
        Args:
            source (str/int): 입력 소스
            output_dir (str): 출력 디렉토리
            save_txt (bool): 텍스트 결과 저장 여부
            save_conf (bool): 신뢰도 저장 여부
        """
        self.source = source
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.cam_width = cam_width
        self.cam_height = cam_height
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 컴포넌트 초기화
        print("=" * 60)
        print("AI 객체 감지 및 트래킹 시스템 초기화")
        print("=" * 60)
        
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.visualizer = Visualizer()
        self.fps_counter = FPSCounter()
        
        # 통계 변수
        self.total_detections = 0
        self.class_counts = defaultdict(int)
        self.detection_history = []
        
        print("초기화 완료!")
        print("=" * 60)
    
    def process_image(self, image_path):
        """
        이미지 파일 처리
        
        Args:
            image_path (str): 이미지 파일 경로
        """
        print(f"이미지 처리 중: {image_path}")
        
        # 이미지 로드
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return
        
        # 객체 감지
        detections = self.detector.detect(frame)
        
        # 감지 결과 시각화
        result_frame = self.detector.draw_detections(frame, detections)
        
        # 정보 패널 추가
        result_frame = self.visualizer.draw_info_panel(
            result_frame, 
            fps=0, 
            frame_count=1,
            detection_count=len(detections)
        )
        
        # 워터마크 추가
        result_frame = self.visualizer.add_watermark(result_frame)
        
        # 결과 저장
        output_path = os.path.join(self.output_dir, f"result_{Path(image_path).name}")
        cv2.imwrite(output_path, result_frame)
        print(f"결과 저장: {output_path}")
        
        # 텍스트 결과 저장
        if self.save_txt:
            self._save_detections_txt(detections, output_path.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # 결과 표시
        if Config.SHOW_DISPLAY:
            cv2.imshow('AI Detection Result', result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_video(self):
        """
        비디오/웹캠 처리
        """
        # 출력 경로 설정
        output_path = None
        if Config.SAVE_VIDEO:
            output_path = create_output_path(self.source, "tracked")
        
        # 비디오 프로세서 초기화 (캠 해상도 전달)
        video_processor = VideoProcessor(self.source, output_path, desired_width=self.cam_width, desired_height=self.cam_height)
        
        print(f"비디오 처리 시작: {self.source}")
        if output_path:
            print(f"출력 저장: {output_path}")
        
        try:
            while True:
                # 프레임 읽기
                ret, frame = video_processor.read_frame()
                if not ret:
                    print("비디오 처리 완료 또는 프레임 읽기 실패")
                    break
                
                # FPS 업데이트
                self.fps_counter.update()
                
                # 객체 감지
                detections = self.detector.detect(frame)
                
                # 객체 트래킹
                tracks = self.tracker.update(detections, frame)
                
                # 통계 업데이트
                self._update_statistics(detections)
                
                # 결과 시각화
                result_frame = self._visualize_results(frame, detections, tracks, video_processor)
                
                # 프레임 저장
                if Config.SAVE_VIDEO:
                    video_processor.write_frame(result_frame)
                
                # 텍스트 결과 저장
                if self.save_txt:
                    frame_info = video_processor.get_frame_info()
                    txt_path = os.path.join(self.output_dir, f"frame_{frame_info['frame_count']:06d}.txt")
                    self._save_tracks_txt(tracks, txt_path)
                
                # 결과 표시
                if Config.SHOW_DISPLAY:
                    cv2.imshow('AI Detection & Tracking', result_frame)
                    
                    # 키 입력 처리
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("사용자에 의해 종료됨")
                        break
                    elif key == ord('r'):
                        print("트래커 리셋")
                        self.tracker.reset()
                    elif key == ord('s'):
                        # 현재 프레임 스크린샷 저장
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        screenshot_path = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
                        cv2.imwrite(screenshot_path, result_frame)
                        print(f"스크린샷 저장: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다.")
        
        except Exception as e:
            print(f"처리 중 오류 발생: {e}")
        
        finally:
            # 리소스 해제
            video_processor.release()
            print("처리 완료!")
            
            # 최종 통계 출력
            self._print_final_statistics()
    
    def _visualize_results(self, frame, detections, tracks, video_processor):
        """
        결과 시각화
        
        Args:
            frame (numpy.ndarray): 원본 프레임
            detections (list): 감지 결과
            tracks (list): 트래킹 결과
            video_processor (VideoProcessor): 비디오 프로세서
            
        Returns:
            numpy.ndarray: 시각화된 프레임
        """
        # 감지 결과 그리기
        result_frame = self.detector.draw_detections(frame, detections, show_confidence=True)
        
        # 트래킹 결과 그리기
        result_frame = self.tracker.draw_tracks(result_frame, tracks, 
                                              show_trajectory=True, show_id=True)
        
        # 정보 패널 그리기
        frame_info = video_processor.get_frame_info()
        result_frame = self.visualizer.draw_info_panel(
            result_frame,
            fps=self.fps_counter.get_fps(),
            frame_count=frame_info['frame_count'],
            track_count=self.tracker.get_track_count(),
            detection_count=len(detections),
            elapsed_time=self.fps_counter.get_elapsed_time()
        )
        
        # 진행률 바 (비디오 파일인 경우)
        if frame_info['total_frames'] > 0:
            result_frame = self.visualizer.draw_progress_bar(
                result_frame, 
                frame_info['progress'],
                frame_info['total_frames']
            )
        
        # 감지 통계 (상위 5개 클래스만)
        top_classes = dict(sorted(self.class_counts.items(), 
                                key=lambda x: x[1], reverse=True)[:5])
        if top_classes:
            result_frame = self.visualizer.draw_detection_stats(result_frame, top_classes)
        
        # 워터마크 추가
        result_frame = self.visualizer.add_watermark(result_frame)
        
        return result_frame
    
    def _update_statistics(self, detections):
        """
        통계 정보 업데이트
        
        Args:
            detections (list): 감지 결과
        """
        self.total_detections += len(detections)
        
        for detection in detections:
            class_id = detection[5]
            class_name = self.detector.get_class_name(class_id)
            self.class_counts[class_name] += 1
        
        # 감지 이력 저장 (최근 1000개만 유지)
        self.detection_history.extend(detections)
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
    
    def _save_detections_txt(self, detections, output_path):
        """
        감지 결과를 텍스트 파일로 저장
        
        Args:
            detections (list): 감지 결과
            output_path (str): 출력 파일 경로
        """
        with open(output_path, 'w') as f:
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                class_name = self.detector.get_class_name(class_id)
                
                if self.save_conf:
                    f.write(f"{class_name} {confidence:.6f} {x1} {y1} {x2} {y2}\n")
                else:
                    f.write(f"{class_name} {x1} {y1} {x2} {y2}\n")
    
    def _save_tracks_txt(self, tracks, output_path):
        """
        트래킹 결과를 텍스트 파일로 저장
        
        Args:
            tracks (list): 트래킹 결과
            output_path (str): 출력 파일 경로
        """
        with open(output_path, 'w') as f:
            for track in tracks:
                x1, y1, x2, y2, track_id, class_id, confidence = track
                class_name = self.detector.get_class_name(class_id)
                
                if self.save_conf:
                    f.write(f"{track_id} {class_name} {confidence:.6f} {x1} {y1} {x2} {y2}\n")
                else:
                    f.write(f"{track_id} {class_name} {x1} {y1} {x2} {y2}\n")
    
    def _print_final_statistics(self):
        """
        최종 통계 출력
        """
        print("\n" + "=" * 60)
        print("최종 처리 통계")
        print("=" * 60)
        print(f"총 감지 수: {self.total_detections}")
        print(f"총 처리 시간: {self.fps_counter.get_elapsed_time():.1f}초")
        print(f"평균 FPS: {self.fps_counter.get_fps():.1f}")
        
        if self.class_counts:
            print("\n클래스별 감지 수:")
            for class_name, count in sorted(self.class_counts.items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count}")
        
        print("=" * 60)

def parse_arguments():
    """
    명령행 인수 파싱
    """
    parser = argparse.ArgumentParser(description='AI 객체 감지 및 트래킹 시스템')
    
    parser.add_argument('--source', type=str, default='0',
                       help='입력 소스 (0: 웹캠, 파일 경로)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='출력 디렉토리')
    
    parser.add_argument('--save-txt', action='store_true',
                       help='텍스트 결과 저장')
    
    parser.add_argument('--save-conf', action='store_true',
                       help='신뢰도 포함하여 저장')
    
    parser.add_argument('--conf-thres', type=float, default=Config.CONFIDENCE_THRESHOLD,
                       help='신뢰도 임계값')
    
    parser.add_argument('--iou-thres', type=float, default=Config.IOU_THRESHOLD,
                       help='IoU 임계값')
    
    parser.add_argument('--no-display', action='store_true',
                       help='화면 출력 비활성화')
    
    parser.add_argument('--no-save', action='store_true',
                       help='비디오 저장 비활성화')

    # Camera resolution options for webcam
    parser.add_argument('--cam-width', type=int, default=None, help='웹캠 캡처 너비 (픽셀)')
    parser.add_argument('--cam-height', type=int, default=None, help='웹캠 캡처 높이 (픽셀)')
    
    return parser.parse_args()

def main():
    """
    메인 함수
    """
    args = parse_arguments()
    
    # 설정 업데이트
    Config.CONFIDENCE_THRESHOLD = args.conf_thres
    Config.IOU_THRESHOLD = args.iou_thres
    Config.SHOW_DISPLAY = not args.no_display
    Config.SAVE_VIDEO = not args.no_save
    
    # 입력 소스 처리
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # AI 시스템 초기화
    try:
        ai_system = AIDetectionTracker(
            source=source,
            output_dir=args.output_dir,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            cam_width=args.cam_width,
            cam_height=args.cam_height
        )
        
        # 입력 타입에 따른 처리
        if isinstance(source, str) and is_image_file(source):
            ai_system.process_image(source)
        else:
            ai_system.process_video()
    
    except Exception as e:
        print(f"시스템 초기화 또는 실행 중 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
