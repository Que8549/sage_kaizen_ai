@echo off
setlocal EnableExtensions

set "ROOT=F:\Projects\sage_kaizen_ai"
set "EXE=%ROOT%\llama.cpp\build\bin\Release\llama-server.exe"
set "MODEL=E:\DeepSeek-V3.2-GGUF\UD-IQ1_S\DeepSeek-V3.2-UD-IQ1_S-00001-of-00004.gguf"
set "LOGDIR=%ROOT%\logs"

REM Streamlit/manager log (your existing file)
set "MGRLOG=%LOGDIR%\q5_server.log"
REM Dedicated llama-server log (NEW)
set "LLAMALOG=%LOGDIR%\q5_llama.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

>>"%MGRLOG%" echo ==== Q5 START %DATE% %TIME% ====
>>"%MGRLOG%" echo EXE=%EXE%
>>"%MGRLOG%" echo MODEL=%MODEL%
>>"%MGRLOG%" echo LLAMALOG=%LLAMALOG%

if not exist "%EXE%" (
  >>"%MGRLOG%" echo ERROR: EXE not found: %EXE%
  exit /b 1
)

if not exist "%MODEL%" (
  >>"%MGRLOG%" echo ERROR: MODEL not found: %MODEL%
  exit /b 1
)

REM IMPORTANT: Use llama-server native logging (more reliable than cmd redirection)
start "sage-q5-llama-server" /D "%ROOT%" /B /MIN cmd.exe /C ^
""%EXE%" ^
  --host 127.0.0.1 ^
  --port 8011 ^
  --model "%MODEL%" ^
  --alias UD-IQ1_S ^
  --ctx-size 4096 ^
  --batch-size 96 ^
  --ubatch-size 48 ^
  --cpu-moe ^
  --device CUDA0,CUDA1 ^
  --split-mode layer ^
  --tensor-split 3,1 ^
  --main-gpu 0 ^
  --n-gpu-layers auto ^
  --fit on ^
  --flash-attn auto ^
  --log-colors off --log-timestamps --log-prefix --log-verbosity 3 ^
  --log-file "%LLAMALOG%" ^
  1> NUL 2>&1"

endlocal
