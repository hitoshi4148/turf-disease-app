@echo off
REM ===============================
REM 芝生病害分類AI Streamlit起動バッチ
REM ===============================

REM 仮想環境をアクティベート
call venv\Scripts\activate.bat

REM Streamlit アプリ起動
streamlit run app.py

REM ===============================
REM このウィンドウは Streamlit 終了まで開いたまま
REM ===============================
pause