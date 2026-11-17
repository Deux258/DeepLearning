@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist ".runtime\rocm\Scripts\python.exe" (
    echo Falta el entorno ROCm. Sigue la seccion Entorno del README de DeepLearning.
    popd
    pause
    exit /b 1
)

set "cuaderno=%~1"
if not defined cuaderno set "cuaderno=entrenamiento.ipynb"
if not defined HF_HOME set "HF_HOME=%~dp0.runtime\hf"
".runtime\rocm\Scripts\python.exe" -m notebook "%cuaderno%"
set "estado=%errorlevel%"
popd
if not "%estado%"=="0" pause
exit /b %estado%
