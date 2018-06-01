
import os
from numpy import dtype, shape
from tensorflow.python.framework.dtypes import string
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import collections

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


def _parse_function(record):
    features = {
        'terms' : tf.VarLenFeature(dtype=tf.string),
        'labels' : tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }
    
    parsed_features = tf.parse_single_example(record, features)
    
    terms = parsed_features['terms'].values
    labels = parsed_features['labels']
    
    return {'terms': terms}, labels

def _input_fn(input_filenames, batch_size=1, num_epochs=None, shuffle=True):
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)
    
    if shuffle:
        ds = ds.shuffle(buffer_size = 10000)
        
    ds = ds.padded_batch(batch_size, ds.output_shapes)
    ds = ds.repeat(num_epochs)
    feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
    return feature_batch, label_batch

def construct_feature_columns():
    feature_column = tf.feature_column.categorical_column_with_vocabulary_list('terms', informative_terms)
    return [tf.feature_column.embedding_column(feature_column, 4)]
    

def train_model(learning_rate, steps, batch_size, hidden_units, train_path, test_path):
    periods = 10
    steps_per_period = steps / periods
    
    training_input_fn = lambda: _input_fn(train_path, batch_size)
    test_input_fn = lambda: _input_fn(test_path, batch_size)
       
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    classifier = tf.estimator.DNNClassifier(feature_columns=construct_feature_columns(), hidden_units=hidden_units, optimizer=optimizer)
    
    print ('Training model...')
    print ('Log Loss (on training data):')
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        
        classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        
        train_evaluation_metrics = classifier.evaluate(input_fn=training_input_fn, steps=steps_per_period)
        test_evaluation_metrics = classifier.evaluate(input_fn=test_input_fn, steps=steps_per_period)
        
        print('Training auc_precision_recall: {} Test auc_precision_recall: {}', train_evaluation_metrics['auc_precision_recall'], test_evaluation_metrics['auc_precision_recall'] )

    print("Model training finished.")
    
    print('Training set metrics: ')
    for m in train_evaluation_metrics:
        print(m, train_evaluation_metrics[m])
    print('---')
    
    print('Test set metrics: ')
    for m in test_evaluation_metrics:
        print(m, test_evaluation_metrics[m])
    print('---')
            
    return classifier


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

informative_terms = None

terms_file_path = os.path.join(os.path.dirname(__file__), 'data/terms.txt')

with open(terms_file_path) as f:
    informative_terms = list(set(f.read().split()))

ds = tf.data.TFRecordDataset(train_path)
ds = ds.map(_parse_function)

classifier = train_model(learning_rate=0.003, steps=2000, batch_size=30, hidden_units=[20,20], train_path=train_path, test_path=test_path)

print(classifier.get_variable_names())

embedding_weights = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')
layer_0_weights = classifier.get_variable_value('dnn/hiddenlayer_0/kernel')
layer_1_weights = classifier.get_variable_value('dnn/hiddenlayer_1/kernel')

print('Embedding weights shape:{}', embedding_weights.shape)
print('Layer 0 weights shape:{}', layer_0_weights.shape)
print('Layer 1 weights shape:{}', layer_1_weights.shape)

'''for term_index in range(len(informative_terms)):
    term_vector = np.zeros(len(informative_terms))
    term_vector[term_index] = 1
    
    embedding_xy = np.matmul(term_vector, embedding_weights)

    plt.text(embedding_xy[0], embedding_xy[1], informative_terms[term_index])
plt.rcParams['figure.figsize'] = (15,15)
plt.xlim(1.2 * embedding_weights.min(), 1.2 * embedding_weights.max())
plt.ylim(1.2 * embedding_weights.min(), 1.2 * embedding_weights.max())
plt.show()

print(ds)
print(ds.output_shapes)

example = ds.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    print(sess.run(example))


minst_dataframe = minst_dataframe.reindex(np.random.permutation(minst_dataframe.index))

training_targets, training_examples = parse_labels_and_features(minst_dataframe.head(7500))
validation_targets, validation_examples = parse_labels_and_features(minst_dataframe.tail(2500))


classifier = train_model(learning_rate=0.01, steps=10, batch_size=10, training_examples=training_examples, training_targets=training_targets, validation_examples=validation_examples, validation_targets=validation_targets)
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_examples.hist(bins=20, figsize=(18,12), xlabelsize=10)
plt.show()
scale_params = get_linear_scale_params_df(training_examples)
z_score_params = get_z_score_params_df(training_examples)

linear_scale_normalize_fn = lambda x, feature, scale_params=scale_params: linear_scale(x, feature, scale_params)
z_score_normalize_fn = lambda x, feature, z_score_params=z_score_params: z_score_normalize(x, feature, z_score_params)

normalize_fn = z_score_normalize_fn

training_examples = normalize(training_examples, z_score_normalize_fn)
training_examples.hist(bins=20, figsize=(18,12), xlabelsize=10)
plt.show()

training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = normalize(preprocess_features(california_housing_dataframe.tail(5000)), normalize_fn)
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

dnn_regressor = train_model(optimizer=tf.train.AdamOptimizer(learning_rate=0.007), steps=2000, batch_size=100, hidden_units=[10,10], training_examples=training_examples, training_targets=training_targets, validation_examples=validation_examples, validation_targets=validation_targets)

california_housing_test_data = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")

test_examples = normalize(preprocess_features(california_housing_test_data), normalize_fn)
test_targets = preprocess_targets(california_housing_test_data)
predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets, shuffle=False, num_epochs=1)

test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = predictions_to_numpy_array(test_predictions)
test_rmse = calculate_root_mean_squared_error(test_predictions, test_targets)
print("Root Mean squared error for the test set: {:.3f} ".format(test_rmse))'''

