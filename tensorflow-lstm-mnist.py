
import os
from sklearn.tests.test_multiclass import n_classes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

time_steps = 28
num_units=128
n_input=28
learning_rate=0.001
n_classes=10
batch_size=128

out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes])) 
print(mnist)