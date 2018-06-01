import os
from _operator import length_hint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from setuptools.dist import Feature
from tensorflow.python.ops.metrics_impl import root_mean_squared_error,\
    false_positives, true_positives
from idlelib.pyparse import trans
from pandas.core.algorithms import quantile

def preprocess_features(california_housing_dataframe):
    
    selected_features = california_housing_dataframe[
        ['latitude', 
         'longitude', 
         'housing_median_age', 
         'total_rooms',
         'total_bedrooms',
         'population',
         'households',
         'median_income' ]]
    processed_features = selected_features.copy()
    processed_features['rooms_per_person'] = processed_features['total_rooms'] / processed_features['population']
    return processed_features

def preprocess_targets(california_housing_dataframe, quantile):
    output_targets = pd.DataFrame()
    output_targets['median_house_value_is_high'] = (california_housing_dataframe['median_house_value'] > quantile).astype(float)
    return output_targets

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None): 
    features = {key: np.array(value) for key, value in dict(features).items()}
    temp = (features,targets)
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size = 1000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(feature) for feature in input_features])

def predictions_to_numpy_array(predictions):
    return np.array([item['probabilities'] for item in predictions])

def calculate_root_mean_squared_error(predictions, targets):
    return math.sqrt(metrics.mean_squared_error(predictions, targets))
    
def calculate_log_loss(predictions, targets):
    log_loss = 0
    print(targets)
    print(targets[0])
    for i in range(len(predictions)):
         log_loss += -1*targets[i]*math.log(predictions[i]) - (1-targets[i])*math.log(1-predictions[i])
         
    return log_loss

def print_0_values(predictions):
    count = 0
    for i in range(len(predictions)):
         if predictions[i] <= 0:
            count+=1
         
    print('Zero Count: {}'.format(count))  

def calculate_log_loss2(predictions, targets):
    return metrics.log_loss(targets, predictions)

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps/periods
       
    my_label = 'median_house_value_is_high'
   
    feature_columns = construct_feature_columns(training_examples)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    linear_classifier = tf.estimator.LinearClassifier(feature_columns = feature_columns, optimizer=optimizer)
    
    training_input_fn=lambda:my_input_fn(training_examples, training_targets[my_label], batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets[my_label], shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets[my_label], shuffle=False, num_epochs=1)
    
    print ('Training model...')
    print ('Log Loss (on training data):')
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = linear_classifier.predict(input_fn=predict_training_input_fn)
        validation_predictions = linear_classifier.predict(input_fn=predict_validation_input_fn)
        evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
        training_predictions = predictions_to_numpy_array(training_predictions)
        validation_predictions = predictions_to_numpy_array(validation_predictions)
        validation_probabilities = np.array([item[1] for item in validation_predictions])
        training_log_loss2 = calculate_log_loss2(training_predictions, training_targets)
        validation_log_loss2 = calculate_log_loss2(validation_predictions, validation_targets)
        training_log_losses.append(training_log_loss2)
        validation_log_losses.append(validation_log_loss2)
        #print("Log Loss for period {} training set: {:.3f} validation set: {:.3f}".format(period, training_log_loss, validation_log_loss))
        print("Log Loss2 for period {} training set: {:.3f} validation set: {:.3f}".format(period, training_log_loss2, validation_log_loss2))
        print("Evaluation metrics for period {} auc: {:.3f} accuracy: {:.3f}".format(period, evaluation_metrics['auc'], evaluation_metrics['accuracy']))
        #print("Log Loss2 for period {} validation set: {:.3f}".format(period, validation_log_loss2))
        
        
    print("Model training finished.")
    
    plt.ylabel('LogLoss')
    plt.xlabel('Periods')
    plt.title('LogLoss per Periods')
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.show()
    return linear_classifier


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

quantile = california_housing_dataframe['median_house_value'].quantile(0.75)
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000), quantile)

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000), quantile)

linear_classifier = train_model(learning_rate = 0.000001, steps=3000, batch_size=10, training_examples=training_examples, training_targets=training_targets, validation_examples=validation_examples, validation_targets=validation_targets)

california_housing_test_data = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")

test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data, quantile)
predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets, shuffle=False, num_epochs=1)

test_predictions = linear_classifier.predict(input_fn=predict_test_input_fn)
test_predictions = predictions_to_numpy_array(test_predictions)
test_log_loss = calculate_log_loss2(test_predictions, test_targets)
print("Log loss for the test set: {:.3f} ".format(test_log_loss))

test_probabilities = np.array([item[1] for item in test_predictions])

false_positives, true_positives, thresholds = metrics.roc_curve(test_targets, test_probabilities)

plt.plot(false_positives, true_positives, label='our_model')
plt.show()

