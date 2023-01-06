import copy, numpy as np
import scipy.io as io
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 隐层节点数
K = 20

# sigmoid函数
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# sigmoid导数
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    a = np.copy(x)  # 复制一份，防止原数据被修改
    a[a <= 0] = 0
    a[a > 0] = 1
    return a

def rnn(u, w, v, x_train,cal_loss):
    alpha = 0.001  # 学习率
    o_deltas = list()  # 输出层的误差
    s_values = list()  # 第一层的值（隐含状态）
    du = 0
    dv = 0
    dw = 0
    msesum = 0
    s_values.append(np.zeros((K, 1)))  # 第一个隐含状态需要0作为它的上一个隐含状态
    # 前向传播
    for i in range(x_train.shape[1]-1):
        x = x_train[:, i].reshape(3, 1)
        # s = sigmoid(np.dot(w, s_values[-1]) + np.dot(u, x))  # (20,1)
        s = np.tanh(np.dot(w, s_values[-1]) + np.dot(u, x))
        s_values.append(copy.deepcopy(s))
        o = np.dot(v, s)  # (3,1)
        # 计算误差损失
        if cal_loss == True:
            dmse = mean_squared_error(x_train[:, i+1], o)
            msesum += dmse
        # 记录反向传播数据
        e = x_train[:, i+1].reshape(3, 1) - o
        layer_delta2 = e  # 线性激活函数
        o_deltas.append(copy.deepcopy(layer_delta2))
    future_s_delta = np.zeros((K, 1))
    # 反向传播
    for i in range(x_train.shape[1]-1):
        x = x_train[:, x_train.shape[1]-i-2].reshape(3, 1)
        prev_s = s_values[-i - 2]
        s = s_values[-i - 1]
        layer_delta2 = o_deltas[-i - 1]
        # layer_delta1 = np.multiply(np.add(np.dot(w.T, future_s_delta), np.dot(v.T, layer_delta2)),sigmoid_output_to_derivative(s))
        layer_delta1 = np.multiply(np.add(np.dot(w.T, future_s_delta), np.dot(v.T, layer_delta2)), (1-s**2))
        du += np.dot(layer_delta1, x.T)
        dw += np.dot(layer_delta1, prev_s.T)
        future_s_delta = layer_delta1
    dv = np.dot(o_deltas[- 1],s_values[- 1].T)
    u = u + alpha * du
    w = w + alpha * dw
    v = v + alpha * dv
    avgloss = msesum/(x_train.shape[1]-1)

    return u,w,v,avgloss


if __name__ == "__main__":
    df = io.loadmat('POI.mat')  # (3,2427)的数据，用于训练的每一个x维度为3x1，时序有2000+
    # print(df.keys())   # 用于输出信息，查找所需数据
    data = df['SquPOI']  # 用于训练与测试的数据.
    # 划分训练与测试
    N = 1000
    X_train = data[:,0:N]
    X_test = data[:,N:-1]
    # 初始化权重, 保证输入x为3x1，s为3x1,输出o为3x1
    rd = np.random.RandomState(12)
    u = np.array(2 * rd.rand(K, 3) - 1)
    w = np.array(2 * rd.rand(K, K) - 1)
    v = np.array(2 * rd.rand(3, K) - 1)
    # 初始化s_initial，作为第一个状态之前的s
    s_initial = np.zeros((K, 1))
    # 训练500轮
    train_loss_values=[]
    for i in range(300):
        print(f'当前为第{i+1}轮训练\n')
        u, w, v,train_loss = rnn(u, w, v, data,cal_loss=True)
        train_loss_values.append(train_loss)
        # print('loss为：',train_loss)
    # 计算预测结果
    x_pred = np.zeros((3,1))
    # x_pred = np.array([[]]*3)
    s_before = s_initial
    mse_sum = 0
    for j in range(X_test.shape[1]-1):
        x = X_test[:, j].reshape(3, 1)
        # s = sigmoid(np.dot(u, x)+np.dot(w, s_before))
        s = np.tanh(np.dot(u, x) + np.dot(w, s_before))
        o = np.dot(v, s)
        temp = np.array(o)
        x_pred = np.hstack((x_pred,temp))
        s_before = s
        d_mse = mean_squared_error(X_test[:, j+1], o)
        mse_sum += d_mse
        # print(f'第{j+1}个数据的mse',d_mse)
        # mse_sum += (X_test[:,j+1].reshape(-1,1) - o)**2
        new_x_train = data[:, 0:N + j + 1]
        for i in range(1):
            u, w, v,temploss = rnn(u, w, v, new_x_train,cal_loss=False)
    mse = mse_sum/(X_test.shape[1]-1)
    print('mse为：', mse)
    # 绘制训练loss曲线
    xx = range(len(train_loss_values))
    plt.plot(xx,train_loss_values)
    plt.show()
    """# 定义图像和三维格式坐标轴
    figdata = x_pred
    fig = plt.figure()
    ax1 = Axes3D(fig)
    xd = figdata[0,:]
    yd = figdata[1,:]
    zd = figdata[2,:]
    ax1.scatter3D(xd, yd, zd, cmap='Blues',s=5)  # 绘制散点图
    # ax1.plot3D(x, y, z, 'gray')  # 绘制空间曲线
    plt.show()"""
