import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



w1 = tf.Variable(tf.ones([1,10]))
w2 = tf.Variable(tf.ones([784,10]))
w3 = tf.Variable(tf.ones([10]))
# w2 = tf.Variable(tf.random_normal([2,3],dtype=tf.float64,stddev=1),name="w2")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(w1))