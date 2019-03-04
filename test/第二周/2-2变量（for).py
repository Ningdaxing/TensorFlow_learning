import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#创建一个变量初始化为0

state = tf.Variable(0,name='conunter1')
#创建一个op，作用使state加1
new_value = tf.add(state,1)
#赋值op
update = tf.assign(state,new_value)  #将后面的值赋值给前面的值

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for i in range(5):
        sess.run(update)
        print(sess.run(state))