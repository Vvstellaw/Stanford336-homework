@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PROJECT_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
set "TIKTOKEN_CACHE_DIR=%PROJECT_ROOT%.tools\tiktoken-cache"
set "TEMP=%PROJECT_ROOT%.tools\tmp"
set "TMP=%PROJECT_ROOT%.tools\tmp"
if not exist "%PROJECT_PYTHON%" (
    echo Project Python is missing. Run scripts\setup_windows.ps1 first.
    exit /b 1
)
pushd "%PROJECT_ROOT%"
if /i "%~1"=="check" goto check
if /i "%~1"=="test" goto test
if /i "%~1"=="test-all" goto all
echo Usage: project.cmd check ^| test ^| test-all
set "PROJECT_EXIT=2"
goto finish
:check
"%PROJECT_PYTHON%" scripts\check_environment.py
set "PROJECT_EXIT=%ERRORLEVEL%"
goto finish
:test
"%PROJECT_PYTHON%" scripts\run_completed_tests.py
set "PROJECT_EXIT=%ERRORLEVEL%"
goto finish
:all
"%PROJECT_PYTHON%" scripts\cache_reference_tokenizer.py --offline
if errorlevel 1 (
    set "PROJECT_EXIT=1"
    goto finish
)
"%PROJECT_PYTHON%" -m pytest tests -q -p no:cacheprovider
set "PROJECT_EXIT=%ERRORLEVEL%"
:finish
popd
exit /b %PROJECT_EXIT%
