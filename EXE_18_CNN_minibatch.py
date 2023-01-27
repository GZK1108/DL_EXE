import scipy.io as io
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate2d


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    return ex / sum(ex)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    a = np.copy(x)  # 复制一份，防止原数据被修改
    a[a <= 0] = 0
    a[a > 0] = 1
    return a


def dropout(y, ratio):
    ym = (np.random.rand(*y.shape) < (1-ratio)) / (1-ratio)
    return ym


def cnn(w1, w3, w4, X, D):
    alpha = 0.02  # 学习率
    k = 4  # 设置权重更新频率，当k=1，等效于sgd
    for nn in range(0, X.shape[2], k):
        temp_w1 = np.zeros_like(w1)
        temp_w3 = np.zeros_like(w3)
        temp_w4 = np.zeros_like(w4)
        for i in range(nn, nn + k):  # 第i个训练样本
            x = X[:, :, i]  # 28x28
            d = D[:, i].reshape(-1, 1)
            # 前向训练
            # 卷积层，valid卷积
            v1 = np.zeros((24, 24, 10))  # 此处直接计算v1的大小，生成对应大小的零矩阵作为初值
            for n in range(10):
                v1[:, :, n] = correlate2d(x, w1[:, :, n], 'valid')  # 卷积层，滤波
            y1 = relu(v1)
            # 池化层，采用平均池化，每4个取均值
            Y2 = (y1[0::3,0::3]+y1[0::3,1::3]+y1[0::3,2::3]+y1[1::3,0::3]+y1[1::3,1::3]+y1[1::3,2::3]+y1[2::3,0::3]+y1[2::3,1::3]+y1[2::3,2::3])/9  # 8x8x10
            y2 = Y2.reshape(-1, 1)  # 将数据排列成一列
            # 全连接层
            v3 = np.dot(w3, y2)
            y3 = np.tanh(v3)
            v4 = np.dot(w4, y3)
            y = softmax(v4)
            # 反向传播
            e = d - y  # 偏差
            delta = e  # 交叉熵+softmax，10x1
            e3 = np.dot(w4.T, delta)
            delta3 = (1-y3**2) * e3
            e2 = np.dot(w3.T, delta3)
            E2 = e2.reshape(Y2.shape)  # 还原成为三维矩阵
            E1 = np.zeros(y1.shape)  # 生成零矩阵，便于存储E1值
            E2_9 = E2 / 9
            E1[0::3, 0::3, :] = E2_9
            E1[0::3, 1::3, :] = E2_9
            E1[0::3, 2::3, :] = E2_9
            E1[1::3, 0::3, :] = E2_9
            E1[1::3, 1::3, :] = E2_9
            E1[1::3, 2::3, :] = E2_9
            E1[2::3, 0::3, :] = E2_9
            E1[2::3, 1::3, :] = E2_9
            E1[2::3, 2::3, :] = E2_9
            delta1 = relu_derivative(v1) * E1
            # 计算w1更新量
            dw1 = np.zeros_like(w1)  # 生成修改量
            for m in range(10):
                dw1[:, :, m] = alpha * correlate2d(x, delta1[:, :, m], 'valid')  # 滤波
                temp_w1[:,:,m] = temp_w1[:,:,m] + w1[:,:,m] + dw1[:,:,m]
            # 计算w3更新量
            temp_w3 = temp_w3 + w3 + alpha * np.dot(delta3, y2.T)
            # 计算w4更新量
            temp_w4 = temp_w4 + w4 + alpha * np.dot(delta, y3.T)
        w1 = temp_w1 / k
        w3 = temp_w3 / k
        w4 = temp_w4 / k
    return w1, w3, w4


if __name__ == "__main__":
    epochs = 1  # 训练轮数
    data = io.loadmat('MNISTData.mat')
    # print(data.keys())
    # 读取训练与测试数据
    D_Train = data['D_Train']
    D_Test = data['D_Test']
    X_Train = data['X_Train']
    X_Test = data['X_Test']

    # 设置初始参数
    rd = np.random.RandomState(42)
    w1 = np.array(2 * rd.rand(5, 5, 10) - 1)
    w3 = np.array((2 * rd.rand(64, 640) - 1) / 10)
    w4 = np.array((2 * rd.rand(10, 64) - 1) / 10)
    # print(X_Test.shape[2])

    # 统计耗时
    sum_time = 0
    for i in range(epochs):
        print(f'当前进行第{i + 1}轮训练')
        start_time = time.time()
        w1, w3, w4 = cnn(w1, w3, w4, X_Train, D_Train)
        end_time = time.time()
        sum_time = sum_time + end_time - start_time
    print('单轮训练平均用时：', sum_time / epochs)
    print('训练完成，测试数据\n')
    # 测试数据
    true = 0  # 用于统计正确分类的个数
    for i in range(X_Test.shape[2]):  # 第i个训练样本
        x = X_Test[:, :, i]  # 28x28
        d = D_Test[:, i].reshape(-1, 1)
        v1 = np.zeros((24, 24, 10))  # 此处直接计算v1的大小，生成对应大小的零矩阵作为初值
        for n in range(10):
            v1[:, :, n] = correlate2d(x, w1[:, :, n], 'valid')  # 卷积层，滤波
        y1 = relu(v1)
        # 池化层，采用平均池化，每4个取均值
        Y2 = (y1[0::3,0::3]+y1[0::3,1::3]+y1[0::3,2::3]+y1[1::3,0::3]+y1[1::3,1::3]+y1[1::3,2::3]+y1[2::3,0::3]+y1[2::3,1::3]+y1[2::3,2::3])/9
        y2 = Y2.reshape(-1, 1)  # 将数据排列成一列
        # 全连接层
        v3 = np.dot(w3, y2)
        y3 = np.tanh(v3)
        v4 = np.dot(w4, y3)
        y = softmax(v4)  # 10x1
        maxindex = np.argmax(y)
        if d[maxindex] != 0:
            true = true + 1
    print('正确率为：', true / X_Test.shape[2])
