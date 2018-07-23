from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range



def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions,1) == np.argmax(labels, 1))/predictions.shape[0])

pickle_file = './data/notMNIST.pickle'

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
  
train_subset = 10000
image_size = 28
num_labels = 10
num_steps = 801
batch_size = 128
hidden_nodes = 1024

graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape = (None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    weights_hidden = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_nodes]))
    biases_hidden = tf.Variable(tf.zeros([hidden_nodes]))
    logits_hidden = tf.matmul(tf_train_dataset, weights_hidden) + biases_hidden
    relu_hidden = tf.nn.relu(logits_hidden)
    
    weights_output = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    biases_output = tf.Variable(tf.zeros([num_labels]))
    logits_output = tf.matmul(relu_hidden, weights_output) + biases_output
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_output))
    
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    predict = tf.nn.softmax(logits_output)
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
            tf_train_labels:batch_labels
        }
        _,l, predictions, weights_h, weights_o = sess.run([optimizer, loss, predict, weights_hidden,weights_output], feed_dict = feed_dict)
        
        #print(weights_h)
        #print('--------------------------------------------------')
        #print(weights_o)
        #print('--------------------------------------------------')
        
        if (step % 100 == 0):
            print('Loss at step {}: {:.2f}'.format(step, l));
            train_accuracy = accuracy(predictions, batch_labels)
            print('Training accuracy: {:.2f}%'.format(train_accuracy))
            valid_accuray = accuracy(sess.run(predict, {tf_train_dataset:valid_dataset}), valid_labels)
            print('Validation accuracy: {:.2f}%'.format(valid_accuray))
    
    test_accuray = accuracy(sess.run(predict, {tf_train_dataset:test_dataset}), test_labels)
    print('Test accuracy: {:.2f}%'.format(test_accuray))