@echo off
setlocal

set ROOT=F:\Projects\sage_kaizen_ai
set EXE=F:\Projects\sage_kaizen_ai\llama.cpp\build\bin\Release\llama-server.exe
set PATH=F:\Projects\sage_kaizen_ai\llama.cpp\build\bin\Release;%PATH%
set LOGDIR=%ROOT%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOGFILE=%LOGDIR%\q5_server.log

set MODEL=E:\DeepSeek-V3.2-GGUF\UD-IQ1_S\DeepSeek-V3.2-UD-IQ1_S-00001-of-00004.gguf

echo ==== Q5 START %DATE% %TIME% ====>> "%LOGFILE%"

REM IMPORTANT: use cmd.exe /C so redirection applies to llama-server itself
start "sage-q5-llama-server" /B /MIN cmd.exe /V:ON /C ^
  ""%EXE%" ^
    --host 127.0.0.1 ^
    --port 8011 ^
    --model "%MODEL%" ^
    --ctx-size 8192 ^
    --batch-size 256 ^
    --ubatch-size 128 ^
    --n-gpu-layers 61 ^
    --split-mode row ^
    --tensor-split 0.60,0.40 ^
    --flash-attn auto ^
    --log-file "%LOGFILE%" ^
    --log-colors off ^
    --log-timestamps ^
    --log-prefix ^
    --log-verbosity 4 ^
  1>>"%LOGFILE%" 2>>&1"
"%EXE%" --version >> "%LOGFILE%" 2>>&1

endlocal
