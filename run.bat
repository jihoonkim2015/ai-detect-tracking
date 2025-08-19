@echo off
echo ===============================================
echo AI 객체 감지 및 트래킹 시스템 실행
echo ===============================================

echo.
echo 1. 웹캠으로 실시간 감지 및 트래킹
echo 2. 데모 실행 (대화형 메뉴)
echo 3. 종료
echo.

set /p choice=옵션을 선택하세요 (1-3): 

if "%choice%"=="1" (
    echo.
    echo 웹캠 실행 중... (종료하려면 'q'를 누르세요)
    python main.py --source 0 --save-txt --save-conf
) else if "%choice%"=="2" (
    echo.
    echo 데모 실행 중...
    python demo.py
) else if "%choice%"=="3" (
    echo.
    echo 프로그램을 종료합니다.
    exit
) else (
    echo.
    echo 잘못된 선택입니다.
)

pause
