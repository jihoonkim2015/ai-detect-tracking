@echo off
echo ===============================================
echo AI 객체 감지 및 트래킹 시스템 설치 스크립트
echo ===============================================

echo.
echo Python 및 pip 버전 확인...
python --version
pip --version

echo.
echo 필요한 패키지 설치 중...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

echo.
echo 설치 완료!
echo.
echo 실행 방법:
echo   웹캠: python main.py --source 0
echo   비디오: python main.py --source path/to/video.mp4
echo   이미지: python main.py --source path/to/image.jpg
echo   데모: python demo.py
echo.

pause
