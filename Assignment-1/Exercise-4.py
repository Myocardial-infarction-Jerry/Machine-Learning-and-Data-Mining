import numpy as np
import random
import matplotlib.pyplot as plt

N = 20000
X = Y = np.linspace(1, 7, 7)
LEFT = (-1, 0)
RIGHT = (1, 0)
UP = (0, -1)
DOWN = (0, 1)


def ant():  # Simulating ant walking
    pos = (1, 1)
    visited = set()
    mid = True
    while True:
        visited.add(pos)
        next_pos = []
        for direction in [LEFT, RIGHT, UP, DOWN]:
            pos_ = tuple(np.array(pos) + np.array(direction))
            if (pos_[0] in X and pos_[1] in Y and pos_ not in visited):
                next_pos.append(pos_)
            else:
                if (pos_ == (4, 4) and mid):
                    mid = False
                    next_pos.append(pos_)
        if len(next_pos) == 0:
            return False
        pos = random.choice(next_pos)
        if (pos == (7, 7)):
            return True


count = 0
for it in range(N):
    if ant():
        count += 1
P = count / N
print('Possibility = %.8lf' % (P))
