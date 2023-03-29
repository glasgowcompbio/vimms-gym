import matplotlib.pyplot as plt
import numpy as np


def apex_reward(distance, scaling_factor=2.0):
    """
    Calculate the penalty for a given distance from the apex using an exponential function.

    Args:
        distance: The distance from the apex, in the range [0, 1].
        scaling_factor: A positive scaling factor controlling the penalty for suboptimal fragmentation times.

    Returns:
        float: The penalty, in the range [0, 1].
    """
    # reward = 1 - distance
    # reward = 1 - distance ** 2
    reward = 1 - np.exp(-scaling_factor * distance)
    return reward


# Generate an array of distances from 0 to 1
distances = np.linspace(0, 1, num=100)

# Calculate the penalties for different scaling factors
scaling_factors = [1.0, 2.0, 3.0, 4.0, 10, 20]
rewards = {}
for sf in scaling_factors:
    rewards[sf] = [apex_reward(d, scaling_factor=sf) for d in distances]

# Plot the penalties for different scaling factors
plt.figure(figsize=(10, 6))
for sf in scaling_factors:
    plt.plot(distances, rewards[sf], label=f"Scaling factor: {sf}")

plt.xlabel("Distance from apex")
plt.ylabel("Reward")
plt.legend()
plt.title("Apex reward for different scaling factors")
plt.grid()
plt.show()
