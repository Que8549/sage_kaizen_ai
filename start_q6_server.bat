@echo off
setlocal

set ROOT=F:\Projects\sage_kaizen_ai
set EXE=F:\Projects\sage_kaizen_ai\llama.cpp\build\bin\Release\llama-server.exe
set PATH=F:\Projects\sage_kaizen_ai\llama.cpp\build\bin\Release;%PATH%
set LOGDIR=%ROOT%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOGFILE=%LOGDIR%\q6_server.log

set MODEL=E:\DeepSeek-V3.2-GGUF\UD-IQ1_M\DeepSeek-V3.2-UD-IQ1_M-00001-of-00005.gguf

echo ==== Q6 START %DATE% %TIME% ====>> "%LOGFILE%"

start "sage-q6-llama-server" /B /MIN cmd.exe /V:ON /C ^
  ""%EXE%" ^
    --host 127.0.0.1 ^
    --port 8012 ^
    --model "%MODEL%" ^
    --ctx-size 8192 ^
    --batch-size 192 ^
    --ubatch-size 96 ^
    --n-gpu-layers 55 ^
    --split-mode row ^
    --tensor-split 0.60,0.40 ^
    --flash-attn auto ^
    --log-file "%LOGFILE%" ^
    --log-colors off ^
    --log-timestamps ^
    --log-prefix ^
    --log-verbosity 4 ^
  1>>"%LOGFILE%" 2>>&1"

endlocal
