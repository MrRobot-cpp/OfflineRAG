@echo off
echo Enhanced RAG Manager - Add Files and Query
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the enhanced RAG manager
echo Starting Enhanced RAG Manager...
python scripts\rag_manager.py

pause
