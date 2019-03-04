import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Fetch是在一个会话中同时运行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mu1 = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([mu1,add])
    print(result)

###############################################

#feed的数据以字典的形式传入
input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output = tf.multiply(input4,input5)

with tf.Session() as sess:

    print(sess.run(output,feed_dict={input4:[7.],input5:[2.]}))