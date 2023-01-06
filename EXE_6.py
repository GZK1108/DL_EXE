# 练习6 使用动量法替代BP算法
import time

import numpy as np
from matplotlib import pyplot as plt

N = 4  # 设置隐层节点数
beta = 0.9  # 设置动量法中历史更新值影响影响大小


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def mmt(w1, w2, x, d):
    w2 = w2.reshape(1, N)
    alpha = 0.9  # 设置步长
    mmt1 = np.full((N, 3), 0)  # 生成零矩阵
    mmt2 = np.full((1, N), 0)
    for i in range(4):
        x_t = x[i, :].reshape(1, 3).T
        v1 = np.dot(w1, x[i, :])
        y1 = sigmoid(v1)
        y1_t = y1.reshape(1, v1.shape[0]).T
        v = np.dot(w2, y1_t)
        y = sigmoid(v)
        e = d[i] - y
        delta = y * (1 - y) * e  # 计算输出层的delta
        # 开始反向传播
        e1 = delta * w2
        delta1 = y1 * (1 - y1) * e1  # 生成1x4数列，不是矩阵
        # 计算第1层更新
        delta1_t = delta1.reshape(1, N).T
        dw1 = np.dot(alpha * delta1_t, x[i, :].reshape(1, 3))
        mmt1 = dw1 + beta * mmt1
        w1 = w1 + mmt1
        # 计算第2层更新
        dw2 = alpha * delta * y1
        mmt2 = dw2 + beta * mmt2
        w2 = w2 + mmt2
    return w1, w2


if __name__ == "__main__":
    # start = time.time()
    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    d = np.array([0, 1, 1, 0])
    epochs = 2000  # 训练2000轮
    # 设置随机数种子
    rd = np.random.RandomState(42)
    w1 = np.array(2 * rd.rand(N, 3) - 1)
    w2 = np.array(2 * rd.rand(1, N) - 1)
    print('初始权重为\n')
    print('w1:', w1)
    print('w2:', w2)
    train_y = []
    for i in range(epochs):
        e2 = 0
        for j in range(4):
            v1 = np.dot(w1, x[j, :].T)
            y1 = sigmoid(v1)
            y1_t = y1.reshape(1, v1.shape[0]).T  # 转置
            v = np.dot(w2, y1_t)
            y = sigmoid(v)
            e2 = e2 + (d[j] - y) ** 2  # 计算误差
            # if e2 < 0.01:
            # print('达到设定标准，训练轮数为：', i+1)
        w1, w2 = mmt(w1, w2, x, d)
        train_y.extend(e2)
    # 计算输出
    res = []
    e2 = 0
    for i in range(4):
        v1 = np.dot(w1, x[i, :].T)
        y1 = sigmoid(v1)
        v = np.dot(w2, y1.T)
        y = sigmoid(v)
        e2 = e2 + (d[i] - y) ** 2  # 计算误差
        res.extend(y)
    print('最终权重为\n')
    print('w1:', w1)
    print('w2:', w2, '\n')
    print('最终输出结果：', res)
    print('误差为：', e2)
    # end = time.time()
    # print('用时为：',end - start)
    train_x = range(1, epochs + 1)
    plt.plot(train_x, train_y)
    plt.xlabel('rounds')
    plt.ylabel('error')
    plt.title(f'beta={beta}')
    plt.show()
