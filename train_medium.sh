#!/bin/bash

env_name="DDAEnv"
model_name="DQN"
timesteps=20E6
repeat=1
verbose=2
preset="QCB_resimulated_medium"
alphas=( 0.00 0.25 0.50 0.75 1.00 )
betas=( 0.50 )
horizons=( 1 4 )

for (( i=0; i<${repeat}; i++ ))
do
  results="notebooks/${preset}/${i}"
  for alpha in "${alphas[@]}"
  do
    for beta in "${betas[@]}"
    do
      for horizon in "${horizons[@]}"
      do
        out_file="${env_name}_${model_name}_alpha_${alpha}_beta_${beta}_horizon_${horizon}.zip"
        echo "Training ${out_file}"
#        python training.py --preset {preset} --model ${model_name} --timesteps ${timesteps} --results ${results} --verbose ${verbose} --out_file ${out_file} --horizon ${horizon}
      done
    done
  done
done