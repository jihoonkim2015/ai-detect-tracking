"""
간단한 데모 실행 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path

def run_demo():
    """
    데모 실행
    """
    print("=" * 60)
    print("AI 객체 감지 및 트래킹 시스템 데모")
    print("=" * 60)
    
    print("\n사용 가능한 옵션:")
    print("1. 웹캠으로 실시간 감지 및 트래킹")
    print("2. 샘플 비디오 처리 (비디오 파일이 있는 경우)")
    print("3. 이미지 처리 (이미지 파일이 있는 경우)")
    print("4. 종료")
    
    while True:
        try:
            choice = input("\n옵션을 선택하세요 (1-4): ").strip()
            
            if choice == '1':
                print("\n웹캠 데모를 시작합니다...")
                print("종료하려면 'q'를 누르세요.")
                print("트래커 리셋: 'r'")
                print("스크린샷 저장: 's'")

                # 해상도 선택 메뉴
                try:
                    res_choice = input("해상도를 선택하세요 (1: 640x480, 2: 1280x720, 3: 1920x1080, 4: Custom, Enter: 기본): ").strip()
                    cam_w = None
                    cam_h = None
                    if res_choice == '1':
                        cam_w, cam_h = 640, 480
                    elif res_choice == '2':
                        cam_w, cam_h = 1280, 720
                    elif res_choice == '3':
                        cam_w, cam_h = 1920, 1080
                    elif res_choice == '4':
                        cam_w = int(input('원하는 너비(px)를 입력하세요: ').strip())
                        cam_h = int(input('원하는 높이(px)를 입력하세요: ').strip())
                except Exception:
                    cam_w = None
                    cam_h = None

                cmd = [
                    sys.executable, "main.py",
                    "--source", "0",
                    "--save-txt",
                    "--save-conf"
                ]
                if cam_w:
                    cmd.extend(["--cam-width", str(cam_w)])
                if cam_h:
                    cmd.extend(["--cam-height", str(cam_h)])

                subprocess.run(cmd)
                
            elif choice == '2':
                # 비디오 파일 찾기
                video_files = []
                data_dir = Path("data/videos")
                
                if data_dir.exists():
                    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                        video_files.extend(data_dir.glob(f"*{ext}"))
                
                if not video_files:
                    print("\n비디오 파일이 없습니다.")
                    print("data/videos/ 폴더에 비디오 파일을 넣어주세요.")
                    continue
                
                print("\n사용 가능한 비디오 파일:")
                for i, video_file in enumerate(video_files):
                    print(f"{i+1}. {video_file.name}")
                
                try:
                    video_choice = int(input("비디오를 선택하세요: ")) - 1
                    if 0 <= video_choice < len(video_files):
                        video_path = str(video_files[video_choice])
                        print(f"\n비디오 처리 시작: {video_path}")
                        
                        subprocess.run([
                            sys.executable, "main.py",
                            "--source", video_path,
                            "--save-txt",
                            "--save-conf"
                        ])
                    else:
                        print("잘못된 선택입니다.")
                        continue
                        
                except ValueError:
                    print("숫자를 입력해주세요.")
                    continue
                
            elif choice == '3':
                # 이미지 파일 찾기
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.extend(Path('.').glob(f"*{ext}"))
                    image_files.extend(Path('data').glob(f"*{ext}"))
                
                if not image_files:
                    print("\n이미지 파일이 없습니다.")
                    print("프로젝트 폴더에 이미지 파일을 넣어주세요.")
                    continue
                
                print("\n사용 가능한 이미지 파일:")
                for i, image_file in enumerate(image_files):
                    print(f"{i+1}. {image_file.name}")
                
                try:
                    image_choice = int(input("이미지를 선택하세요: ")) - 1
                    if 0 <= image_choice < len(image_files):
                        image_path = str(image_files[image_choice])
                        print(f"\n이미지 처리 시작: {image_path}")
                        
                        subprocess.run([
                            sys.executable, "main.py",
                            "--source", image_path,
                            "--save-txt",
                            "--save-conf"
                        ])
                    else:
                        print("잘못된 선택입니다.")
                        continue
                        
                except ValueError:
                    print("숫자를 입력해주세요.")
                    continue
                
            elif choice == '4':
                print("데모를 종료합니다.")
                break
                
            else:
                print("잘못된 선택입니다. 1-4 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n\n데모가 중단되었습니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_demo()
