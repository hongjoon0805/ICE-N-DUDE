#! /bin/bash

python3 BW_channel_estimate.py --t 0 &
python3 BW_channel_estimate.py --t 1
python3 ~/send_email.py BW

