import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pandas as pd

ITER_TIME = 100
X_min, X_max = 2, 4
Y_min, Y_max = -1, 1


def f(x, y):
    return (np.power(y, 2) * np.exp(-np.power(y, 2)) + np.power(x, 4) *
            np.exp(-np.power(x, 2))) / (x * np.exp(-np.power(x, 2)))


X = np.linspace(X_min, X_max, 100)
Y = np.linspace(Y_min, Y_max, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
Z_min, Z_max = Z.min(), Z.max()

fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('ln(Z)')

surf = ax.plot_surface(X,
                       Y,
                       np.log(Z),
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False,
                       alpha=0.3)

plt.savefig(os.path.dirname(__file__) + '/Exercise-3_Surface.png', dpi=300)

N_list = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 500, 5000])

datas = pd.DataFrame()

for N in N_list:
    I = np.empty(0)
    for it in range(ITER_TIME):
        X = np.random.rand(N) * (X_max - X_min) + X_min
        Y = np.random.rand(N) * (Y_max - Y_min) + Y_min
        Z = np.random.rand(N) * Z_max
        position = np.where(Z < f(X, Y), 1,
                            0)  # Calculating random points inplace
        inplace = position.sum()
        I = np.append(I,
                      inplace / N * (X_max - X_min) * (Y_max - Y_min) * Z_max)
    datas = pd.concat([
        datas,
        pd.Series({
            'N': N,
            'Average': np.mean(I),
            'Variance': np.var(I)
        }).to_frame().T
    ])
    if N == N_list[-1]:

        position = np.where(position, 'green', 'red')
        ax.scatter3D(X, Y, np.log(Z), color=position, alpha=0.7, s=1)

        plt.savefig(os.path.dirname(__file__) + '/Exercise-3.png', dpi=300)
        print(datas)
        # plt.show()
