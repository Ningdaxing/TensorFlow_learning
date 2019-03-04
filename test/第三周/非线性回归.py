import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#使用numpy生成200个随机点(样本)，[:,np.newaxis]是增加一个维度
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
#生成一些干扰值，形状与x_data形状一样
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#定义两个placeholder(先定义形状）根据样本定义,等待喂入数据
x = tf.placeholder(tf.float32,[None,1]) #喂入的数据,如果用常量来做测试数据，每次计算都有新增一个节点，图会变得特别大
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络的中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))  #偏向值
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  #非线性转换,让输出为非线性的，整个模型就是非线性的了

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    #获得预测值，假定模型已经训练好了，输入测试值得到结果
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)#lw为宽度，x_data为输入值（x轴），prediction_value为输出值(y轴)
    plt.show()