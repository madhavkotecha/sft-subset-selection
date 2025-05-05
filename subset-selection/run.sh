#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 manage.py --model=meta-llama/Llama-3.2-3B --method=full --train_seed=50 > cache/logs/experiment1.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 python3 manage.py --model=meta-llama/Meta-Llama-3-8B --method=random --train_seed=50 > cache/logs/experiment2.log 2>&1 &
PID2=$!
CUDA_VISIBLE_DEVICES=2 python3 manage.py --model=meta-llama/Llama-3.2-3B --method=initial --train_seed=50 > cache/logs/experiment3.log 2>&1 &
PID3=$!
CUDA_VISIBLE_DEVICES=3 python3 manage.py --model=meta-llama/Meta-Llama-3-8B --method=delift-se --train_seed=50 > cache/logs/experiment4.log 2>&1 &
PID4=$!

wait $PID1
echo "Run 1 finished."
wait $PID2
echo "Run 2 finished."
wait $PID3
echo "Run 3 finished."
wait $PID4
echo "Run 4 finished."