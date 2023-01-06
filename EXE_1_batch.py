import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 批量，每一轮更新一次w
def bitch(w, x, d):
    alpha = 0.9
    temp_w = np.array([0, 0, 0])
    for i in range(4):
        v = np.dot(w, x[i, :].T)
        y = sigmoid(v)
        e = d[i] - y
        delta = y * (1 - y) * e
        dw = alpha * delta * x[i, :]
        temp_w = temp_w + w + dw.T
    mean_w = temp_w / len(d)
    return mean_w


if __name__ == "__main__":
    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    d = np.array([0, 0, 1, 1])
    rounds = 1000  # 训练1000轮
    # 设置随机数种子
    rd = np.random.RandomState(42)
    w = np.array(rd.randint(-1, 1, (1, 3)))
    print('初始权重为：', w)
    train_y = []
    for i in range(rounds):
        e2 = 0
        for j in range(4):
            v = np.dot(w, x[j, :].T)
            y = sigmoid(v)
            e2 = e2 + (d[j] - y) ** 2  # 计算误差
        w = bitch(w, x, d)
        train_y.extend(e2)
    # 计算输出
    res = []
    for i in range(4):
        v = np.dot(w, x[i, :].T)
        y = sigmoid(v)
        res.extend(y)
    print('最终权重：', w)
    print('最终输出结果：', res)
    train_x = range(1, rounds + 1)
    plt.plot(train_x, train_y)
    plt.xlabel('rounds')
    plt.ylabel('error')
    plt.title('bitch')
    plt.show()
