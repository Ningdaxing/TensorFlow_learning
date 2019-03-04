import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#载入数据集,直接填写MNIST_data，会放到当前目录下。执行这条语句会去网上找
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

#每个批次的大小,一次性放入100张图片(以矩阵形式放入)
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.ones([784,10]))
b = tf.Variable(tf.ones([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#在使用softmax或者S型曲线时用交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#正确率大小，结果存放在一个布尔型列表中。argmax中后面那个1参数，
# axis = 0 的时候返回每一列最大值的位置索引
# axis = 1 的时候返回每一行最大值的位置索引，argmax返回一维向量最大值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率,转换类型 bool --> float32, 再求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size) #数据xs,标签ys
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})  #将训练好的模型，喂入测试数据
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))