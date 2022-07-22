#!/bin/bash

cd simulated_chems_large
python training.py

cd ../QCB_chems_large
python training.py