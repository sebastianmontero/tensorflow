import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

with tf.name_scope('MyOperationGroup'):
    with tf.name_scope('ScopeA'): 
        a = tf.add(1, 2, name='Add_these_numbers')
        b = tf.multiply(a, 3)
    with tf.name_scope('ScopeB'):
        c = tf.add(4, 5, 'And_these')
        d = tf.multiply(c, 6, 'Multiply_these_numbers')
        
with tf.name_scope('ScopeC'):
    e = tf.multiply(4, 5, 'B_add')
    f = tf.div(c, 6, 'B_mul')
    
g = tf.add(b,d)
h = tf.multiply(g,f)

with tf.Session() as session:
    writer = tf.summary.FileWriter('output', session.graph)
    print(session.run(h))
    writer.close()