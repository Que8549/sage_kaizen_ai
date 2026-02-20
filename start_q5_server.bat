@echo off
:: ============================================================================
:: MANUAL USE ONLY
:: Python (server_manager.py) no longer reads this file.
:: All server config is in config/brains/brains.yaml.
:: Use this script only for manual debugging from a terminal.
:: ============================================================================
setlocal EnableExtensions

set "ROOT=F:\Projects\sage_kaizen_ai"
set "EXE=%ROOT%\llama.cpp\build\bin\Release\llama-server.exe"
set "MODEL=E:\Qwen\Qwen2.5-14B-Instruct-GGUF\Q6_K\qwen2.5-14b-instruct-q6_k-00001-of-00004.gguf"
set "LOGDIR=%ROOT%\logs"
set "LOGFILE=%LOGDIR%\q5_server.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

>>"%LOGFILE%" echo ==== Q5 START %DATE% %TIME% ====
>>"%LOGFILE%" echo EXE=%EXE%
>>"%LOGFILE%" echo MODEL=%MODEL%

if not exist "%EXE%" (
  >>"%LOGFILE%" echo ERROR: EXE not found: %EXE%
  exit /b 1
)

if not exist "%MODEL%" (
  >>"%LOGFILE%" echo ERROR: MODEL not found: %MODEL%
  exit /b 1
)

>>"%LOGFILE%" echo Launching llama-server on 127.0.0.1:8011 (devices CUDA1) ...

"%EXE%" --host 127.0.0.1 --port 8011 --model "%MODEL%" --alias Qwen2.5-14B-Q5_K_M --device CUDA1 --n-gpu-layers all --split-mode none --ctx-size 4096 --batch-size 1024 --ubatch-size 512 --threads 10 --threads-batch 10 --threads-http 6 --flash-attn on --cache-type-k f16 --cache-type-v f16 --fit on --fit-target 768 --no-warmup --n-predict 256 --log-colors off --log-timestamps --log-prefix --log-verbosity 3 1>>"%LOGFILE%" 2>>&1

set "RC=%ERRORLEVEL%"
>>"%LOGFILE%" echo ==== Q5 EXIT %DATE% %TIME% (rc=%RC%) ====
endlocal & exit /b %RC%
