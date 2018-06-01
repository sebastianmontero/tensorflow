import os
import tensorflow as tf
import numpy as np
from setuptools.dist import Feature

tf.logging.set_verbosity(tf.logging.ERROR)

file_path = os.path.dirname(os.path.realpath(__file__)) + '/data/olympics.csv'

features = tf.placeholder(tf.int32, [3], 'features')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')

printerop = tf.Print(total, [features,country, total], name='printer')

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    with open(file_path) as file:
        next(file)
        for line in file:
            country_name, code, gold, silver, bronze, total = line.strip().split(',')
            gold = int(gold)
            silver = int(silver)
            bronze = int(bronze)
            total = session.run(printerop, feed_dict={features:[gold,silver,bronze], country:country_name})
            print(country_name, total)

