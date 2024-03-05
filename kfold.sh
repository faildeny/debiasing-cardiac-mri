#!/bin/bash
nohup python cardioai/batch_training.py "$@" --kfold > /dev/null 2>&1&
