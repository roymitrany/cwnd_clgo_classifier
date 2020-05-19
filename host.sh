#!/bin/bash
netcat -l 2345
sleep 2
python3 transmitter.py 5000 ${1} ${2} ${3} ${4} ${5}
#python3 sender.py 500 10.0.2.10 ${2} ${3}