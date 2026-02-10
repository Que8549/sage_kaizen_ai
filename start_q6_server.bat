@echo off
setlocal EnableExtensions

set "ROOT=F:\Projects\sage_kaizen_ai"
set "EXE=%ROOT%\llama.cpp\build\bin\Release\llama-server.exe"
set "MODEL=E:\DeepSeek-V3.2-GGUF\UD-IQ1_M\DeepSeek-V3.2-UD-IQ1_M-00001-of-00005.gguf"
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

>>"%LOGFILE%" echo Launching llama-server on 127.0.0.1:8012 (devices CUDA0,CUDA1) ...

"%EXE%" --host 127.0.0.1 --port 8012 --model "%MODEL%" --alias UD-IQ1_M --ctx-size 4096 --batch-size 96 --ubatch-size 48 --cpu-moe --device CUDA0,CUDA1 --split-mode layer --tensor-split 3,1 --main-gpu 0 --n-gpu-layers auto --fit on --flash-attn auto --log-colors off --log-timestamps --log-prefix --log-verbosity 3 1>>"%LOGFILE%" 2>>&1

set "RC=%ERRORLEVEL%"
>>"%LOGFILE%" echo ==== Q6 EXIT %DATE% %TIME% (rc=%RC%) ====
endlocal & exit /b %RC%
