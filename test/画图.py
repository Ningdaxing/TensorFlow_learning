import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
#生成一些干扰值，形状与x_data形状一样
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

plt.figure()
plt.scatter(x_data,y_data)
plt.show()