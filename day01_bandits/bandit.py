import numpy as np

class Bandit:
    def __init__(self):
        self.probs = [0.1, 0.3, 0.8, 0.5]

    def pull(self, arm):
        return 1 if np.random.rand() < self.probs[arm] else 0

bandit = Bandit()

for _ in range(10):
    print(bandit.pull(2))

# Dumb Agent : Always choose random arm.
total_reward = 0
for _ in range(1000):
    arm = np.random.randint(4)
    reward = bandit.pull(arm)
    total_reward+= reward

print(total_reward)