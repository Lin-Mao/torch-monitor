#!/bin/bash

# Unit test of op and mem information

LD_PRELOAD=$(pwd)/../driver/driver.so python ./add.py cpu > ./log

ret=$?
rm ./log

if [ $ret -ne 0 ]; then
    echo "Error"
    exit 1
fi

echo "Success"
