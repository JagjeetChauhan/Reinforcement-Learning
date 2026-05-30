import numpy as np
from .bandit import Bandit

Bandit = Bandit()

Q = np.zeros(4)
N = np.zeros(4)

epsilon = 0.1
total_reward = 0

# Initial exploration
for arm in range(4):
    reward = Bandit.pull(arm)
    N[arm] += 1
    Q[arm] = reward

# Main Learning Loop
for step in range(1000):
    if np.random.rand() < epsilon:
        arm = np.random.randint(4)
    else:
        arm = np.argmax(Q)
    reward = Bandit.pull(arm)
    total_reward += reward

    N[arm] += 1
    Q[arm] += (reward - Q[arm]) / N[arm]

print("Estimated Values:", Q)
print("Times Chosen:", N)
print("Total Reward:", total_reward)