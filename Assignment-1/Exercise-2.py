import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ITER_TIME = 100

N_list = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 1000])

datas = pd.DataFrame()

for N in N_list:
    I = np.empty(0)
    for it in range(ITER_TIME):
        X = np.random.rand(N)
        Y = np.random.rand(N)
        inplace = np.where(Y < X * X * X, 1,
                           0)  # Calculating random points inplace
        I = np.append(I, inplace.sum() / N)
    datas = pd.concat([
        datas,
        pd.Series({
            'N': N,
            'Average': np.mean(I),
            'Variance': np.var(I)
        }).to_frame().T
    ])
    if N == N_list[-1]:
        fig = plt.figure(figsize=(4, 4))
        edge = np.linspace(0, 1, 1000)
        plt.plot(edge, np.power(edge, 3), color='blue')
        plt.scatter(X, Y, color=np.where(inplace, 'green', 'red'), s=10)
        plt.savefig(os.path.dirname(__file__) + '/Exercise-2.png', dpi=300)
        print(datas)
        plt.show()