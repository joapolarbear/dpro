#!/bin/bash
PROJECT_PATH=$(realpath ./)

pip3 install wheel
python3 -m pip install --upgrade build
python3 -m build

### install python deps
pip3 install -r requirements.txt

### install
pip3 uninstall -y dpro && pip3 install $PROJECT_PATH
# cd .. && python3 -c "import dpro;print(dpro.__path__)" && cd dpro

### install dpro command line
REAL_PATH=$(realpath ./dpro_cli)
if [ -f "/usr/bin/dpro_cli" ]; then
    rm /usr/bin/dpro_cli
fi
cp -f $REAL_PATH /usr/bin/
sed -n 3,10p dpro_cli