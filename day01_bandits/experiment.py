import numpy as np
from .bandit import Bandit
bandit = Bandit()

# Dumb Agent : Always choose random arm.
total_reward = 0
for _ in range(1000):
    arm = np.random.randint(4)
    reward = bandit.pull(arm)
    total_reward+= reward

print(total_reward)

# Learning Agent: Store estimated value of each arm.
Q = np.zeros(4) # Q = estimated reward
N = np.zeros(4) # N = times chosen

total_reward_learning = 0
for _ in range(100):
    arm = np.random.randint(4)
    reward = bandit.pull(arm)
    print(reward, type(reward))
    total_reward_learning+= reward

    N[arm] += 1
    Q[arm] = Q[arm] + (
        reward - Q[arm]
    ) / N[arm]

# This is called: Incremental Mean Update It computes: average reward of arm without storing all rewards.
print(Q)
print(N)
print("Learning Reward: ", total_reward_learning)