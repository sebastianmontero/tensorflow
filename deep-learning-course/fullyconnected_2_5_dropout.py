from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


def create_weights_biases(num_inputs, outputs):
    return tf.Variable(tf.truncated_normal([num_inputs, outputs])), tf.Variable(tf.zeros([outputs]))

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
image_size = 28
num_labels = 10
num_steps = 1801
batch_size = 128
hidden_nodes = 1024
beta = 0.3
learning_rate = 0.005

graph = tf.Graph()

with graph.as_default():
    
    tf_keep_probabilty = tf.placeholder(tf.float32)
    
    tf_train_dataset = tf.placeholder(tf.float32, shape = (None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    trainable_vars = []
    
    trainable_vars.append(create_weights_biases(image_size**2, hidden_nodes))
    logits_hidden = tf.matmul(tf_train_dataset, trainable_vars[0][0]) + trainable_vars[0][1]
    relu_hidden = tf.nn.relu(logits_hidden)
    relu_hidden = tf.nn.dropout(relu_hidden, tf_keep_probabilty)
    
    trainable_vars.append(create_weights_biases(hidden_nodes, num_labels))
    logits_output = tf.matmul(relu_hidden, trainable_vars[1][0]) + trainable_vars[1][1]
    
    l2_loss = tf.add_n([tf.nn.l2_loss(trainable[0]) for trainable in trainable_vars]) * beta;
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits_output)) + l2_loss
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
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
            tf_train_labels:batch_labels,
            tf_keep_probabilty: 0.5
        }
        _,l, predictions = sess.run([optimizer, loss, predict], feed_dict = feed_dict)
        
        #print(weights_h)
        #print('--------------------------------------------------')
        #print(weights_o)
        #print('--------------------------------------------------')
        
        if (step % 100 == 0):
            print('Loss at step {}: {:.2f}'.format(step, l));
            train_accuracy = accuracy(predictions, batch_labels)
            print('Training accuracy: {:.2f}%'.format(train_accuracy))
            valid_accuray = accuracy(sess.run(predict, {tf_train_dataset:valid_dataset, tf_keep_probabilty : 1}), valid_labels)
            print('Validation accuracy: {:.2f}%'.format(valid_accuray))
    
    test_accuray = accuracy(sess.run(predict, {tf_train_dataset:test_dataset, tf_keep_probabilty : 1}), test_labels)
    print('Test accuracy: {:.2f}%'.format(test_accuray))