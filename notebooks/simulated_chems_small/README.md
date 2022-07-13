PPO_1 - repeated MS1 penalty + initial reward scheme, linear_schedule(0.001, min_value=0.0003) -- best results
PPO_2 - no repeated MS1 penalty + intensity difference reward, linear_schedule(0.001, min_value=0.0003)
PPO_3 - no repeated MS1 penalty + intensity difference reward, constant 0.0001
PPO_4 - no repeated MS1 penalty + initial reward scheme, linear_schedule(0.001, min_value=0.0001) -- best results
PPO_5 - no repeated MS1 penalty + additive reward scheme, linear_schedule(0.001, min_value=0.0001)
PPO_6 - no repeated MS1 penalty + additive reward scheme, constant 0.0001