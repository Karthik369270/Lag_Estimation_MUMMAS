@echo off
title MUMMAS Data Curation Dashboard
cd /d "%~dp0"
python serve.py --port 8005
pause
