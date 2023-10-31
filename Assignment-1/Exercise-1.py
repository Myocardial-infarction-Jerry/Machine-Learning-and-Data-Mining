import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ITER_TIME = 100

N_list = np.array([20, 50, 100, 200, 300, 500, 1000, 5000])

datas = pd.DataFrame()

for N in N_list:
    pi = np.empty(0)
    for it in range(ITER_TIME):
        X = np.random.rand(N)
        Y = np.random.rand(N)
        d = np.sqrt(np.square(X) + np.square(Y))
        inplace = np.where(d < 1, 1,
                           0).sum()  # Calculating random points inplace
        pi = np.append(pi, inplace / N * 4)
    datas = pd.concat([
        datas,
        pd.Series({
            'N': N,
            'Average': np.mean(pi),
            'Variance': np.var(pi)
        }).to_frame().T
    ])
    if N == N_list[-1]:
        fig = plt.figure(figsize=(4, 4))
        edge = np.linspace(0, np.pi / 2, 1000)
        plt.plot(np.cos(edge), np.sin(edge), color='blue')
        plt.scatter(X, Y, color=np.where(d < 1, 'green', 'red'), s=1)
        plt.savefig(os.path.dirname(__file__) + '/Exercise-1.png', dpi=300)
        print(datas)
        plt.show()
