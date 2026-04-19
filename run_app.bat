@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv venv
)

echo Installing required packages...
"venv\Scripts\python.exe" -m pip install -r requirements.txt

if not exist "model.pkl" (
  echo Training model because model.pkl is missing...
  "venv\Scripts\python.exe" model.py
)

echo Starting FactLens at http://127.0.0.1:5000
"venv\Scripts\python.exe" app.py
