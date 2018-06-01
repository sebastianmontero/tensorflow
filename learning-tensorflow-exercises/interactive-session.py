import tensorflow as tf
import resource
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

session = tf.InteractiveSession()

print("{} KB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

x = tf.constant(np.eye(10000))
y = tf.constant(np.random.randn(10000,300))

print("{} KB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

z = tf.matmul(x,y)

print("{} KB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

z.eval()

print("{} KB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session.close()


