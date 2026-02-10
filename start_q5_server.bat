@echo off
setlocal EnableExtensions

set "ROOT=F:\Projects\sage_kaizen_ai"
set "EXE=%ROOT%\llama.cpp\build\bin\Release\llama-server.exe"
set "MODEL=E:\DeepSeek-V3.2-GGUF\UD-IQ1_S\DeepSeek-V3.2-UD-IQ1_S-00001-of-00004.gguf"
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

>>"%LOGFILE%" echo Launching llama-server on 127.0.0.1:8011 (devices CUDA0,CUDA1) ...

"%EXE%" --host 127.0.0.1 --port 8011 --model "%MODEL%" --alias UD-IQ1_S --ctx-size 4096 --batch-size 256 --ubatch-size 128 --cpu-moe --device CUDA0,CUDA1 --split-mode layer --tensor-split 2,1 --main-gpu 0 --n-gpu-layers all --n-predict 512 --fit on --flash-attn on --log-colors off --log-timestamps --log-prefix --log-verbosity 3 1>>"%LOGFILE%" 2>>&1

set "RC=%ERRORLEVEL%"
>>"%LOGFILE%" echo ==== Q5 EXIT %DATE% %TIME% (rc=%RC%) ====
endlocal & exit /b %RC%
