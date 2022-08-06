#!/bin/bash

cd QCB_chems_small
rm samplers_QCB_small.p
rm -rf results
python training.py

cd ../QCB_chems_medium
rm samplers_QCB_medium.p
rm -rf results
python training.py