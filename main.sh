#!/bin/bash

nohup python run_final.py --experiment 1 > train.log 2>&1 &
echo $! > train.pid
echo "Training started with PID $(cat train.pid)"

# nohup bash your_script.sh > output.log 2>&1 &
# echo $! > your_script.pid
# echo "Script started with PID $(cat your_script.pid)"
# tail -f output.log
