
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

def create_training_input_fn(features, targets, batch_size=1): 
    
    def _input_fn(num_epochs=None, shuffle=True):
        idx = np.random.permutation(features.index)
        raw_features = {'pixels' : features.reindex(idx)}
        raw_targets = np.array(targets[idx])
        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)
        if shuffle:
            ds = ds.shuffle(buffer_size = 1000)
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    
    return _input_fn

def create_predict_input_fn(features, targets, batch_size): 
    
    def _input_fn(num_epochs=None, shuffle=True):
        
        raw_features = {'pixels' : features.values}
        raw_targets = np.array(targets)
        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    
    return _input_fn


def construct_feature_columns():
    informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family")
    return [tf.feature_column.categorical_column_with_vocabulary_list('terms', informative_terms)]


def calculate_root_mean_squared_error(predictions, targets):
        return math.sqrt(metrics.mean_squared_error(predictions, targets))
    
def linear_scale(series, feature, scale_params):
    print(scale_params)
    return series.apply(lambda x: ((x -scale_params[feature]['min_value'])/scale_params[feature]['scale']) - 1)

def log_normalize(series):
    return series.apply(lambda x: math.log(x + 1))

def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x: max(min(x, clip_to_max), clip_to_min))

def z_score_normalize(series, feature, z_score_params):
    return series.apply(lambda x: (x - z_score_params[feature]['mean'])/z_score_params[feature]['std'])

def binary_threshold(series, threshold):
    return series.apply(lambda x: 1 if x > threshold else 0)

def get_linear_scale_params(series):
    min_value = series.min()
    max_value = series.max()
    
    return {
        'min_value': min_value,
        'scale' : (max_value -min_value) / 2
    }

def get_z_score_params(series):
   
    return {
        'mean': series.mean(),
        'std' : series.std()
    }

def get_linear_scale_params_df(dataframe):
    scale_params = {}
    for feature in dataframe:
        scale_params[feature] = get_linear_scale_params(dataframe[feature])
        
    return scale_params

def get_z_score_params_df(dataframe):
    params = {}
    for feature in dataframe:
        params[feature] = get_z_score_params(dataframe[feature])
        
    return params

def normalize(examples, normalize_fn):
    normalized_features = pd.DataFrame()
    
    for feature in examples:
        normalized_features[feature] = normalize_fn(examples[feature], feature)
        
    return normalized_features

def parse_labels_and_features(dataframe):
    labels = dataframe[0]
    features = dataframe.loc[:,1:784]
    features = features / 255
    return labels, features
    

def train_model(learning_rate, steps, batch_size, train_path, test_path):
    periods = 10
    steps_per_period = steps / periods
    
    training_input_fn = lambda: _input_fn(train_path, batch_size)
    test_input_fn = lambda: _input_fn(test_path, batch_size)
       
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    classifier = tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(), optimizer=optimizer)
    
    print ('Training model...')
    print ('Log Loss (on training data):')
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        
        classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        
        train_evaluation_metrics = classifier.evaluate(input_fn=training_input_fn, steps=steps_per_period)
        test_evaluation_metrics = classifier.evaluate(input_fn=test_input_fn, steps=steps_per_period)
        
        print('Training accuracy: {} Test accuracy: {}', train_evaluation_metrics['accuracy'], test_evaluation_metrics['accuracy'] )

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

ds = tf.data.TFRecordDataset(train_path)
ds = ds.map(_parse_function)

classifier = train_model(learning_rate=0.1, steps=1000, batch_size=25, train_path=train_path, test_path=test_path)

'''print(ds)
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

