# 电子科技大学深度学习课程练习


<aside>
💡 OS：windows 10；编程语言：python 3.9，所有内容均为python语言编写

</aside>

---

**练习1**

用SGD、批量和小批量算法，训练网络，给出最终权系数和四个样本的网络输出值【其中，SGD训练1000轮，批量训练4000轮，小批量(2个样本一组)训练2000轮】。

**练习2**

结合课堂练习，比较SGD、批量和小批量三种算法学习速度。说明:每种算法学习1000轮，画出“轮-误差”曲线，其中误差=4个实际输出与期望输出之差的平方和。

**练习3**

用SGD对数据2训练4000轮，给出最终权系数和四个样本的网络输出，验证训练结果是否有效?

**练习4**

作业4:训练浅层NN解决XOR问题。

**练习5**

作业5:尝试改变隐层节点个数(3、5、2? ) ,观察能否解决XOR问题?如何避免不收敛?

**练习6**

用动量算法训练浅层NN求解XOR问题

**练习7**

练习7:分别用交叉嫡和误差平方和代价函数训练同一神经网络求解XOR问题,比较误差-轮曲线。

**练习8**

随堂练习

**练习9**

设计和训练神经网络识别以下五个数字

**练习10**

用训练数据训练网络，用测试数据测试训练结果(注:运行多次观察结果是否变化，思考原因)

**练习11**

尝试构造其它测试数据测试网络

**练习12**

补全上述代码，观察训练结果是否有效。

**练习13**

重复运行多次主函数，观察训练结果是否有差异?思考其中原因和改善方法。

**练习14**

比较两种结构的优劣，结合本例比较两者的训练结果，并对结果进行分析。(提示:ReLU真的好吗?)

**练习15**

补全Dropout相关代码，得到训练结果。

**练习16**

Dropout+ReLU如何实现?

**练习17**

已知兴趣点(POI)历史轨迹，训练RNN预测其下一时刻三维坐标
RNN网络如何设计?
更新策略?
练习17∶完成POI预测RNN网络的训练
>模型训练效果一般

**练习18**

使用CNN完成MNIST数据集训练