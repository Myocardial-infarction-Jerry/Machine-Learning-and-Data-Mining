import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange

current_path = os.path.abspath(__file__)
directory = os.path.dirname(current_path)
os.chdir(directory)

# 加载训练数据
train_data = np.loadtxt('dataForTrainingLogistic.txt')
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

# 加载测试数据
test_data = np.loadtxt('dataForTestingLogistic.txt')
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# 添加偏置项
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]


# 定义逻辑回归模型
class LogisticRegression:

    def __init__(self):
        self.parameters = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, parameters):
        m = len(y)
        h = self.sigmoid(np.dot(X, parameters))
        cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def gradient_ascent(self, X, y, learning_rate, num_iterations):
        m, n = X.shape
        self.parameters = np.zeros(n)
        costs = []

        for it in trange(num_iterations):
            h = self.sigmoid(np.dot(X, self.parameters))
            error = h - y
            gradient = np.dot(X.T, error)
            self.parameters -= learning_rate * gradient

            cost = self.compute_cost(X, y, self.parameters)
            costs.append(cost)

        return costs

    def predict(self, X):
        h = self.sigmoid(np.dot(X, self.parameters))
        predictions = np.round(h)
        return predictions


# 训练逻辑回归模型
logreg = LogisticRegression()
learning_rate = 0.001
num_iterations = 1000
costs = logreg.gradient_ascent(X_train, y_train, learning_rate, num_iterations)

# 绘制目标函数值的变化曲线
plt.plot(range(num_iterations), costs)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Objective Function Value on Each Iteration')
plt.savefig('Prof-2e.png', dpi=300)
plt.show()
