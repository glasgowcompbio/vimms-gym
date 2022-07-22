#!/bin/bash

cd simulated_chems_small
python training.py

cd ../QCB_chems_small
python training.py