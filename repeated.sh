#!/bin/bash
nohup python cardioai/batch_training.py "$@" --repeated > /dev/null 2>&1&
