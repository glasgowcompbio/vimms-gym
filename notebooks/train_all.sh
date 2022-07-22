#!/bin/bash

cd simulated_chems_small
python training.py

cd ../QCB_chems_small
python training.py

cd ../simulated_chems_medium
python training.py

cd ../QCB_chems_medium
python training.py

cd ../simulated_chems_large
python training.py

cd ../QCB_chems_large
python training.py