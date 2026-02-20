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
set "MODEL=E:\Qwen\bartowski-Qwen2.5-32B-Instruct-GGUF\Qwen2.5-32B-Instruct-Q6_K_L.gguf"
set "LOGDIR=%ROOT%\logs"
set "LOGFILE=%LOGDIR%\q6_server.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

>>"%LOGFILE%" echo ==== Q6 START %DATE% %TIME% ====
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

>>"%LOGFILE%" echo Launching llama-server on 127.0.0.1:8012 (devices CUDA0) ...

"%EXE%" --host 127.0.0.1 --port 8012 --model "%MODEL%" --alias Q6_K_L --device CUDA0 --n-gpu-layers all --split-mode none  --ctx-size 8192 --batch-size 768 --ubatch-size 256 --threads 22 --threads-batch 22 --threads-http 8 --flash-attn on --cache-type-k f16 --cache-type-v q8_0 --fit on --fit-target 1024 --fit-ctx 8192 --no-warmup --n-predict 512 --log-colors off --log-timestamps --log-prefix --log-verbosity 3 1>>"%LOGFILE%" 2>>&1

set "RC=%ERRORLEVEL%"
>>"%LOGFILE%" echo ==== Q6 EXIT %DATE% %TIME% (rc=%RC%) ====
endlocal & exit /b %RC%
