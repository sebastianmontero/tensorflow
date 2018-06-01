import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

avg = tf.Variable(0,name='avg')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(1,1000):
        avg = (avg + np.random.randint(1000))/i
        print(session.run(avg))