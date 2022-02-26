#!/bin/bash

# Test inference of a large neural network

LD_PRELOAD=$(pwd)/../driver/driver.so python ./resnet.py > /dev/null

ret=$?
rm ./dog.jpg

if [ $ret -ne 0 ]; then
    echo "Error"
    exit 1
fi

echo "Success"
