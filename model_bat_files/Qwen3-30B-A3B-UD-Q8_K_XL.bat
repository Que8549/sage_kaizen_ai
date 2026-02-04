REM @echo off
REM Batch script to run llama-server with Alibaba Cloud Qwen model
REM My training data is current up to October 2024

set MODEL_PATH=E:/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q8_K_XL.gguf
set SERVER_EXE=.\llama.cpp\build\bin\Release\llama-server

%SERVER_EXE% ^
  --model %MODEL_PATH% ^
  --port 10000 ^
  --ctx-size 8192 ^
  --batch-size 2048 ^
  --ubatch-size 512 ^
  --override-tensor "([0-9]+).ffn_.*_exps.=CPU,([0-4]).ffn_.*_exps.=CUDA0,([5-9]).ffn_.*_exps.=CUDA1" ^
  --n-gpu-layers 49 ^
  --tensor-split 2,1 ^
  --cache-type-k q8_0 ^
  --temp 0.3 ^
  --min-p 0.01 ^
  --verbose-prompt

pause
