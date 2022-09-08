#!/bin/bash

env_name="DDAEnv"
repeat=5
timesteps=5E5
verbose=2
preset="QCB_chems_small"
results="notebooks/${preset}"
model_name="DQN"

### Balanced ###

model_type="Balanced"
alpha=0.50
beta=0.00

for (( i=0; i<=${repeat}; i++ ))
do
    out_file="${env_name}_${model_name}_${model_type}_${i}.zip"    
    echo "Training ${out_file}"
    python training.py --preset QCB_chems_small --model DQN --timesteps ${timesteps} --results ${results} --alpha ${alpha} --beta ${beta} --out_file ${out_file} --verbose {verbose}
done

### Intensity ###

model_type="Intensity"
alpha=0.25
beta=0.00

for (( i=0; i<=${repeat}; i++ ))
do
    out_file="${env_name}_${model_name}_${model_type}_${i}.zip"    
    echo "Training ${out_file}"
    python training.py --preset QCB_chems_small --model DQN --timesteps ${timesteps} --results ${results} --alpha ${alpha} --beta ${beta} --out_file ${out_file} --verbose {verbose}
done

### Coverage ###

model_type="Coverage"
alpha=0.75
beta=0.00

for (( i=0; i<=${repeat}; i++ ))
do
    out_file="${env_name}_${model_name}_${model_type}_${i}.zip"    
    echo "Training ${out_file}"
    python training.py --preset QCB_chems_small --model DQN --timesteps ${timesteps} --results ${results} --alpha ${alpha} --beta ${beta} --out_file ${out_file} --verbose {verbose}
done