import numpy as np

N = 100000
ITER_TIME = 100

P = np.empty(0)
for it in range(ITER_TIME):
    A = np.random.rand(N)
    B = np.random.rand(N)
    C = np.random.rand(N)

    upper = np.where(A < 0.85, 1, 0)
    lower = np.bitwise_and(np.where(B < 0.95, 1, 0), np.where(C < 0.9, 1, 0))
    all = np.bitwise_or(upper, lower)

    P = np.append(P, all.sum() / N)
print('Average = %-15.8lf\nVariance = %-15.8lf' % (np.mean(P), np.var(P)))
