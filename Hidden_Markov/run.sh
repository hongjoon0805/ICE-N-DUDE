#! /bin/bash

python3 loss_check_ICE.py --d 0.10 &
python3 loss_check_ICE.py --d 0.20 &
python3 loss_check_ICE.py --d 0.30 
python3 ~/send_email.py Simple_loss_check
