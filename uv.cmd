@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "UV_EXE=%PROJECT_ROOT%.tools\uv\uv.exe"
if not exist "%UV_EXE%" (
    set "UV_EXE="
    for /f "delims=" %%I in ('where.exe uv.exe 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
)
if not defined UV_EXE (
    echo uv is missing. Run scripts\setup_windows.ps1 first.
    exit /b 1
)
set "UV_CACHE_DIR=%PROJECT_ROOT%.tools\cache"
set "UV_PYTHON_INSTALL_DIR=%PROJECT_ROOT%.tools\managed-python"
set "UV_PYTHON_BIN_DIR=%PROJECT_ROOT%.tools\bin"
set "UV_PYTHON_NO_REGISTRY=1"
set "PYTHONUTF8=1"
set "TIKTOKEN_CACHE_DIR=%PROJECT_ROOT%.tools\tiktoken-cache"
set "TEMP=%PROJECT_ROOT%.tools\tmp"
set "TMP=%PROJECT_ROOT%.tools\tmp"
set "UV_PYTHON=3.12.14"
if exist "%PROJECT_ROOT%.tools\python-3.12.14\python.exe" set "UV_PYTHON=%PROJECT_ROOT%.tools\python-3.12.14\python.exe"
pushd "%PROJECT_ROOT%"
"%UV_EXE%" %*
set "PROJECT_EXIT=%ERRORLEVEL%"
popd
exit /b %PROJECT_EXIT%
