from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
from six.moves.urllib.request import urlretrieve
from six.moves import range
from sklearn.manifold import TSNE

url = 'http://mattmahoney.net/dc/'
filename = 'text8.zip'
data_root = './data'
bytes = 31344016
vocabulary_size = 50000
batch_size = 8

def maybe_download(filename, data_root, expected_bytes):
    filepath = os.path.join(data_root, filename)
    fileurl = url + filename
    if not os.path.exists(filepath):
        print('Downloading file: {} ...'.format(fileurl))
        filename,_ = urlretrieve(fileurl, filepath)
    statinfo = os.stat(filepath)
    if statinfo.st_size == expected_bytes:
        print('Found and verified filename: {}'.format(filepath))
    else:
        print('Unable to verify file: {}, size: {}'.format(filepath, statinfo.st_size))
        raise Exception('Failed to verify file: {}'.format(filepath))
    return filepath  

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    
    for word,_ in count:
        dictionary[word] = len(dictionary)
        
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = 0
def generate_batch(batch_size, skip_window):
    global data_index
    
    span = skip_window * 2 + 1
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)
    
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    for i in range(batch_size):
        buffer_list = list(buffer)
        labels[i, 0] = buffer_list.pop(skip_window)
        batch[i] = buffer_list
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(15,15))
    
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x,y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
            

filepath = maybe_download(filename, data_root, bytes)      
words = read_data(filepath)
print('Number of words: {}'.format(len(words)))

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words: ', count[:10])
print('Sample data: ', data[:10])

print('data:', [reverse_dictionary[di] for di in data[:8]])

for skip_window in [1,2,3]:
    data_index = 0
    batch, labels = generate_batch(batch_size, skip_window)
    print('skip_window: {}'.format(skip_window))
    batch_words = []
    for context in batch:
        context_words = [reverse_dictionary[ci] for ci in context]
        batch_words.append(context_words)
    print('batch:', batch_words)
    print('labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    

batch_size = 128
embedding_size = 128
skip_window = 1
span = skip_window * 2 + 1
valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size, span - 1])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0 , 1.0))
    
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, labels=train_labels, inputs=tf.reduce_sum(embed,1), 
                                   num_sampled=num_sampled, num_classes=vocabulary_size))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001
#num_steps = 2000
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, skip_window)
        feed_dict = {
            train_dataset: batch_data,
            train_labels: batch_labels
        }
        
        _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step {}: {:.2f}'.format( step, average_loss))
            average_loss = 0
            
        if step % 10000 == 0:
            sim = similarity.eval()
            
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to {}: '.format(valid_word)
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '{} {},'.format(log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])
words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)

                
        
        