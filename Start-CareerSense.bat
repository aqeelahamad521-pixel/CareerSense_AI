@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CareerSense

echo.
echo CareerSense
echo Preparing the local application...
echo.

if not exist "app.py" (
    set "FAILURE_MESSAGE=The project is incomplete. Keep this launcher beside app.py and requirements.txt."
    goto :failure
)
if not exist "requirements.txt" (
    set "FAILURE_MESSAGE=The project is incomplete. Keep this launcher beside app.py and requirements.txt."
    goto :failure
)

set "PYTHON_COMMAND="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_COMMAND=py -3.11"
)

if not defined PYTHON_COMMAND (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYTHON_COMMAND=python"
    )
)

if not defined PYTHON_COMMAND (
    set "FAILURE_MESSAGE=Python 3.11 was not found. Install it from https://www.python.org/downloads/ and enable the Python launcher, then double-click this file again."
    goto :failure
)

set "VENV_DIR=%CD%\.venv-windows"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Creating the private Python environment...
    %PYTHON_COMMAND% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        set "FAILURE_MESSAGE=The Python environment could not be created."
        goto :failure
    )
)

"%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    set "FAILURE_MESSAGE=The existing .venv-windows environment is not Python 3.11. Delete that folder and run this launcher again."
    goto :failure
)

set "REQUIREMENTS_HASH="
for /f "tokens=1" %%H in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 'requirements.txt').Hash.ToLower()"') do set "REQUIREMENTS_HASH=%%H"
if not defined REQUIREMENTS_HASH (
    set "FAILURE_MESSAGE=The requirements file could not be checked."
    goto :failure
)

set "STAMP_FILE=%VENV_DIR%\.requirements.sha256"
set "INSTALLED_HASH="
if exist "%STAMP_FILE%" set /p INSTALLED_HASH=<"%STAMP_FILE%"

if /I not "%REQUIREMENTS_HASH%"=="%INSTALLED_HASH%" (
    echo Installing the exact application dependencies ^(first run may take a few minutes^)...
    "%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        set "FAILURE_MESSAGE=Dependencies could not be installed. Check your internet connection and try again."
        goto :failure
    )
    >"%STAMP_FILE%" echo %REQUIREMENTS_HASH%
)

echo Starting CareerSense at http://localhost:8501
echo Keep this window open while using the application. Press Control-C here to stop it.
echo.

"%VENV_PYTHON%" -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    set "FAILURE_MESSAGE=The application stopped with exit code %APP_EXIT_CODE%. Review the messages above for details."
    goto :failure
)

echo.
echo CareerSense stopped.
pause
exit /b 0

:failure
echo.
echo CareerSense could not start.
echo %FAILURE_MESSAGE%
echo.
pause
exit /b 1
