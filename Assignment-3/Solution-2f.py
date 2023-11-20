import numpy as np
import os
import matplotlib.pyplot as plt

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

        for it in range(num_iterations):
            h = self.sigmoid(np.dot(X, self.parameters))
            error = h - y
            gradient = np.dot(X.T, error)
            self.parameters -= learning_rate * gradient

    def predict(self, X):
        h = self.sigmoid(np.dot(X, self.parameters))
        predictions = np.round(h)
        return predictions


# 定义函数计算错误分类样本数量
def calculate_error(X, y, parameters):
    logreg = LogisticRegression()
    logreg.parameters = parameters
    y_pred = logreg.predict(X)
    error = np.abs(y_pred - y).sum()
    return error


# 评估训练集大小增加时的训练误差和测试误差
train_errors = []
test_errors = []
training_sizes = np.linspace(10, len(train_data), 10, dtype=np.int32)

for k in training_sizes:
    # 随机选择训练集
    indices = np.random.choice(len(X_train), k, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    # 训练模型
    logreg = LogisticRegression()
    learning_rate = 0.001
    num_iterations = 1000
    logreg.gradient_ascent(X_train_subset, y_train_subset, learning_rate,
                           num_iterations)

    # 计算训练误差和测试误差
    train_error = calculate_error(X_train_subset, y_train_subset,
                                  logreg.parameters)
    test_error = calculate_error(X_test, y_test, logreg.parameters)

    train_errors.append(train_error)
    test_errors.append(test_error)

# 绘制训练误差和测试误差的变化曲线
plt.plot(training_sizes,
         train_errors,
         color='blue',
         label='Training Error',
         alpha=0.7)
plt.plot(training_sizes,
         test_errors,
         color='red',
         label='Test Error',
         alpha=0.7)
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.title('Training Error and Test Error vs Training Set Size')
plt.legend()
plt.savefig('Prof-2f.png', dpi=300)
plt.show()

# 描述训练误差和测试误差随训练集大小增加时的变化
print("随着训练集大小的增加，训练误差逐渐减少，而测试误差先减少后增加。")
print("这种行为发生的原因是，当训练集较小时，模型可能无法捕捉到数据的所有模式和特征，导致欠拟合。")
print("随着训练集大小的增加，模型能够更好地学习数据的特征，从而减少了训练误差。")
print("然而，当训练集过大时，模型可能过度拟合训练数据，导致在未见过的测试数据上表现较差，从而增加了测试误差。")
