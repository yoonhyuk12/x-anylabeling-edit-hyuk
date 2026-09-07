@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv-cpu"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "APP_EXE=%VENV_DIR%\Scripts\xanylabeling.exe"

if not exist "%PYTHON_EXE%" (
    echo [X-AnyLabeling] First-time setup: creating a Python 3.12 environment...
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

if not exist "%APP_EXE%" (
    echo [X-AnyLabeling] First-time setup: installing required packages...
    "%PYTHON_EXE%" -m pip install -U uv
    if errorlevel 1 goto :error

    "%PYTHON_EXE%" -m uv pip install -e ".[cpu]"
    if errorlevel 1 goto :error
)

echo [X-AnyLabeling] Starting...
start "" "%APP_EXE%"
exit /b 0

:error
echo.
echo Installation or startup failed. Review the message above and try again.
pause
exit /b 1
