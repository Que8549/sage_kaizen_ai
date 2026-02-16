@echo off
setlocal EnableExtensions

set "ROOT=F:\Projects\sage_kaizen_ai"
set "EXE=%ROOT%\llama.cpp\build\bin\Release\llama-server.exe"
set "MODEL=E:\bge-m3-GGUF\FP16\bge-m3-FP16.gguf"
set "LOGDIR=%ROOT%\logs"
set "LOGFILE=%LOGDIR%\embed_server.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

>>"%LOGFILE%" echo ==== Embedding Endpoint START %DATE% %TIME% ====
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

>>"%LOGFILE%" echo Launching llama-server on 127.0.0.1:8020 (devices CUDA0) ...

"%EXE%" --host 127.0.0.1 --port 8020 --model "%MODEL%" --alias bge-m3-embed --embeddings --pooling mean --device CUDA0 --n-gpu-layers all --ctx-size 2048 --batch-size 2048 --ubatch-size 512 --threads 20 --threads-batch 20 --threads-http 8 --flash-attn on --log-colors off --log-timestamps --log-prefix --log-verbosity 3 1>>"%LOGFILE%" 2>>&1

set "RC=%ERRORLEVEL%"
>>"%LOGFILE%" echo ==== Embedding Endpoint EXIT %DATE% %TIME% (rc=%RC%) ====
endlocal & exit /b %RC%
