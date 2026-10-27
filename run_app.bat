@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m streamlit run app.py
goto :eof
:fail
echo Setup failed. Install Python 3.13 and retry. See README.md for manual setup.
pause
