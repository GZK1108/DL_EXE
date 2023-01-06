#  练习15，sigmoid+dropout

import numpy as np
import random

N = 20  # 设置隐层节点数
epochs = 10000  # 训练轮数


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    return ex / sum(ex)


def dropout(y, ratio):
    len1 = y.shape[0]  # 行
    len2 = y.shape[1]
    ym = np.zeros((len1, len2))
    num = round(len1 * len2 * (1 - ratio))  # 非零元素个数
    idx = np.array(random.sample(range(0, len1 * len2), num))  # 生成num个不相同的随机数
    for i in idx:  # 利用除法进行定位
        k = i // len1
        r = i - k * len1
        ym[r, k] = len1 * len2 / num
    return ym


def sigmoiddropuout(w1, w2, w3, w4, X, D):
    alpha = 0.1  # 设置步长
    ratio = 0.2  # 设置dropout屏蔽节点比例
    for i in range(5):
        x = X[:, :, i].reshape(25, 1)
        d = D[:, i].reshape(5, 1)
        v1 = np.dot(w1, x)  # Nx1
        y1 = sigmoid(v1)  # Nx1
        y1 = y1 * dropout(y1, ratio)

        v2 = np.dot(w2, y1)  # Nx1
        y2 = sigmoid(v2)
        y2 = y2 * dropout(y2, ratio)

        v3 = np.dot(w3, y2)  # Nx1
        y3 = sigmoid(v3)
        y3 = y3 * dropout(y3, ratio)

        v = np.dot(w4, y3)  # w4:5xN v:5x1
        y = softmax(v)  # 5x1
        e = d - y
        delta = e  # 计算输出层的delta 5x1
        # 开始反向传播
        e3 = np.dot(w4.T, delta)  # 这里不能使用 * ，需要使用矩阵乘，*为点乘，输出Nx1矩阵
        delta3 = y3 * (1 - y3) * e3  # 点乘 Nx1
        e2 = np.dot(w3.T, delta3)
        delta2 = y2 * (1 - y2) * e2
        e1 = np.dot(w2.T, delta2)
        delta1 = y1 * (1 - y1) * e1
        # 计算第1层更新
        dw1 = np.dot(alpha * delta1, x.T)  # Nx25
        w1 = w1 + dw1
        # 计算第2层更新
        dw2 = np.dot(alpha * delta2, y1.T)  # 20xN
        w2 = w2 + dw2
        # 计算第3层更新
        dw3 = np.dot(alpha * delta3, y2.T)
        w3 = w3 + dw3
        # 第四层
        dw4 = np.dot(alpha * delta, y3.T)
        w4 = w4 + dw4
    return w1, w2, w3, w4


if __name__ == "__main__":
    # 训练集
    X = np.zeros((5, 5, 5))
    X[:, :, 0] = [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0]]
    X[:, :, 1] = [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    X[:, :, 2] = [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
    X[:, :, 3] = [[0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0]]
    X[:, :, 4] = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
    D = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    # 测试集
    X_test = np.zeros((5, 5, 5))
    X_test[:, :, 0] = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]]
    X_test[:, :, 1] = [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
    X_test[:, :, 2] = [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
    X_test[:, :, 3] = [[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]]
    X_test[:, :, 4] = [[0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]]
    D_test = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])
    # 新测试集
    X1 = np.zeros((5, 5, 5))
    X1[:, :, 0] = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 1, 1, 0]]
    X1[:, :, 1] = [[1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    X1[:, :, 2] = [[1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]]
    X1[:, :, 3] = [[1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0]]
    X1[:, :, 4] = [[1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]]
    # 设置随机数种子
    rd = np.random.RandomState(42)
    w1 = np.array(2 * rd.rand(N, 25) - 1)
    w2 = np.array(2 * rd.rand(N, N) - 1)
    w3 = np.array(2 * rd.rand(N, N) - 1)
    w4 = np.array(2 * rd.rand(5, N) - 1)
    """print('初始权重为\n')
    print('w1:', w1)
    print('w2:', w2)
    print('w3:',w3)
    print('w4:',w4)"""

    # 模型训练
    train_y = []
    for i in range(epochs):
        w1, w2, w3, w4 = sigmoiddropuout(w1, w2, w3, w4, X, D)

    # 计算训练输出
    for i in range(5):
        x = X[:, :, i].reshape(25, 1)
        v1 = np.dot(w1, x)  # Nx1
        y1 = sigmoid(v1)  # Nx1
        v2 = np.dot(w2, y1)  # Nx1
        y2 = sigmoid(v2)
        v3 = np.dot(w3, y2)  # Nx1
        y3 = sigmoid(v3)
        v = np.dot(w4, y3)  # w4:5xN v:5x1
        y = softmax(v)  # 5x1
        y = np.around(y, decimals=2)
        print(f'第{i + 1}个样本训练结果为：\n', y)

    # 测试集
    for j in range(5):
        x = X_test[:, :, j].reshape(25, 1)
        v1 = np.dot(w1, x)  # Nx1
        y1 = sigmoid(v1)  # Nx1
        v2 = np.dot(w2, y1)  # Nx1
        y2 = sigmoid(v2)
        v3 = np.dot(w3, y2)  # Nx1
        y3 = sigmoid(v3)
        v = np.dot(w4, y3)  # w4:5xN v:5x1
        y = softmax(v)  # 5x1
        y = np.around(y, decimals=2)
        print(f'第{j + 1}个测试样本结果为：\n', y)
