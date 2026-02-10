@echo off
setlocal EnableExtensions

set "ROOT=F:\Projects\sage_kaizen_ai"
set "EXE=%ROOT%\llama.cpp\build\bin\Release\llama-server.exe"
set "MODEL=E:\DeepSeek-V3.2-GGUF\UD-IQ1_M\DeepSeek-V3.2-UD-IQ1_M-00001-of-00005.gguf"
set "LOGDIR=%ROOT%\logs"

set "MGRLOG=%LOGDIR%\q6_server.log"
set "LLAMALOG=%LOGDIR%\q6_llama.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

>>"%MGRLOG%" echo ==== Q6 START %DATE% %TIME% ====
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

start "sage-q6-llama-server" /D "%ROOT%" /B /MIN cmd.exe /C ^
""%EXE%" ^
  --host 127.0.0.1 ^
  --port 8012 ^
  --model "%MODEL%" ^
  --alias UD-IQ1_M ^
  --ctx-size 4096 ^
  --batch-size 72 ^
  --ubatch-size 36 ^
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
