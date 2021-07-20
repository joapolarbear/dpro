#!/bin/bash

pip3 install -r requirements.txt
REAL_PATH=$(realpath ./dpro)
ln -sf $REAL_PATH /usr/bin/
sed -n 3,10p dpro
