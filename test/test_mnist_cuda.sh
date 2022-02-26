#!/bin/bash

# Test training of a basic neural network

LD_PRELOAD=$(pwd)/../driver/driver.so python ./mnist.py --epochs 1 --dry-run > /dev/null

ret=$?
rm -rf ../data

if [ $ret -ne 0 ]; then
    echo "Error"
    exit 1
fi

echo "Success"
