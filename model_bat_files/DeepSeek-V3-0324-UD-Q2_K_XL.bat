REM @echo off
REM Batch script to run llama-server with DeepSeek-V3-0324-GGUF model

set MODEL_PATH=E:/DeepSeek-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf
set SERVER_EXE=.\llama.cpp\build\bin\Release\llama-server

set LLAMA_CUDA_UNIFIED_MEMORY=1
REM set CUDA_VISIBLE_DEVICES=0,1

%SERVER_EXE% ^
  --model %MODEL_PATH% ^
  --port 10000 ^
  --ctx-size 8192 ^
  --batch-size 2048 ^
  --ubatch-size 512 ^
  --override-tensor "([0-9]+).ffn_.*_exps.=CPU,([0-4]).ffn_.*_exps.=CUDA0,([5-9]).ffn_.*_exps.=CUDA1" ^
  --n-gpu-layers 40 ^
  --tensor-split 2,1 ^
  --cache-type-k q8_0 ^
  --temp 0.3 ^
  --min-p 0.01 ^
  --verbose-prompt

pause
