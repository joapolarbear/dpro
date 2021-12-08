#!/bin/bash

pip3 install -r requirements.txt
REAL_PATH=$(realpath ./dpro_cli)
ln -sf $REAL_PATH /usr/bin/
sed -n 3,10p dpro_cli

pip3 install wheel
python3 -m pip install --upgrade build
python3 -m build

pip3 uninstall -y dpro && pip3 install /home/tiger/ws/git/dpro