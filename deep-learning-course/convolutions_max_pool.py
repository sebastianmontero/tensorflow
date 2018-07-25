from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions,1) == np.argmax(labels, 1))/predictions.shape[0])
        

pickle_file = './data/notMNIST-conv.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
image_size = 28
num_labels = 10
num_steps = 2801
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_channels = 1

beta = 0.3
starting_learning_rate = 0.007

graph = tf.Graph()

with graph.as_default():
    
    tf_keep_probabilty = tf.placeholder(tf.float32)
    
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size, image_size, 1))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros((depth)))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([(image_size // 4)**2 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding = 'SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        hidden = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding = 'SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        hidden = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        hidden = tf.nn.dropout(hidden, keep_prob=tf_keep_probabilty)
        return tf.matmul(hidden, layer4_weights) + layer4_biases
    
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits)) #+ l2_loss
    
    tf_global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starting_learning_rate, tf_global_step, decay_steps=100, decay_rate=0.98) 
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    train_predict = tf.nn.softmax(logits)
    valid_predict = tf.nn.softmax(model(tf_valid_dataset))
    test_predict = tf.nn.softmax(model(tf_test_dataset))
    #valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print('Initialized')
    
    for step in range(num_steps):
        
        offset = (step * batch_size) % (train_dataset.shape[0] - batch_size)
        batch_data = train_dataset[offset: offset + batch_size]
        batch_labels = train_labels[offset: offset + batch_size]
        
        feed_dict = {
            tf_train_dataset:batch_data,
            tf_train_labels:batch_labels,
            tf_keep_probabilty: 0.75,
            tf_global_step: step
        }
        _,l, predictions, lr = sess.run([optimizer, loss, train_predict, learning_rate], feed_dict = feed_dict)
        
        #print(weights_h)
        #print('--------------------------------------------------')
        #print(weights_o)
        #print('--------------------------------------------------')
        
        if (step % 100 == 0):
            print('Loss at step {}: {:.2f}'.format(step, l))
            print('Learning rate: {:.4f}'.format(lr));
            train_accuracy = accuracy(predictions, batch_labels)
            print('Training accuracy: {:.2f}%'.format(train_accuracy))
            valid_accuray = accuracy(sess.run(valid_predict, feed_dict={tf_keep_probabilty :1.0}), valid_labels)
            print('Validation accuracy: {:.2f}%'.format(valid_accuray))
    
    test_accuray = accuracy(sess.run(test_predict, feed_dict={tf_keep_probabilty :1.0}), test_labels)
    print('Test accuracy: {:.2f}%'.format(test_accuray))