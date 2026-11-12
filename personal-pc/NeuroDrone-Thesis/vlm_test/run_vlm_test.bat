@echo off
title NeuroDrone VLM test
cd /d "%~dp0"
echo NeuroDrone - first VLM test through Ollama
echo Run 2: downloads a ~2 GB model, then tests 32 photos on the GPU and again on the CPU (30+ min).
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_vlm_test.ps1" %*
echo.
echo Finished. The results are in the "results" folder next to this file.
pause
