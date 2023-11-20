import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

current_path = os.path.abspath(__file__)
directory = os.path.dirname(current_path)
os.chdir(directory)

# 加载训练数据
train_data = np.loadtxt('dataForTrainingLinear.txt')
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

# 加载测试数据
test_data = np.loadtxt('dataForTestingLinear.txt')
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# 初始化参数
iter_times = 15
num_iterations = 100000

learning_rate = 0.00015
initial_parameters = np.zeros(X_train.shape[1] + 1)


# 定义计算损失函数的函数
def compute_cost(X, y, parameters):
    m = len(y)
    predictions = np.dot(X, parameters[1:]) + parameters[0]
    cost = np.sum((predictions - y)**2) / (2 * m)
    return cost


# 定义梯度下降函数
def gradient_descent(X, y, parameters, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        predictions = np.dot(X, parameters[1:]) + parameters[0]
        parameters[1:] -= (learning_rate / m) * np.dot(X.T, (predictions - y))
        parameters[0] -= (learning_rate / m) * np.sum(predictions - y)
    return parameters


# 进行梯度下降优化
optimal_parameters = [initial_parameters]
for iter in trange(iter_times, desc='Training', unit='epoch'):
    optimal_parameters.append(
        gradient_descent(X_train, y_train, optimal_parameters[-1].copy(),
                         learning_rate, num_iterations))

# 绘制训练误差和测试误差的变化曲线
iterations = np.linspace(0, num_iterations * iter_times, iter_times + 1)
train_errors = [compute_cost(X_train, y_train, _) for _ in optimal_parameters]
test_errors = [compute_cost(X_test, y_test, _) for _ in optimal_parameters]

plt.plot(iterations, train_errors, label='Training Error')
plt.plot(iterations, test_errors, label='Testing Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.savefig('Prof-1a.png', dpi=300)
plt.show()
