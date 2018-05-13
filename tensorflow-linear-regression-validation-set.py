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
from tensorflow.python.ops.metrics_impl import root_mean_squared_error
from idlelib.pyparse import trans

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

def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = california_housing_dataframe['median_house_value'] / 1000
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
    return np.array([item['predictions'][0] for item in predictions])

def calculate_root_mean_squared_error(predictions, targets):
        return math.sqrt(metrics.mean_squared_error(predictions, targets))

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps/periods
       
    my_label = 'median_house_value'
   
    feature_columns = construct_feature_columns(training_examples)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    linear_regressor = tf.estimator.LinearRegressor(feature_columns = feature_columns, optimizer=optimizer)
    
    training_input_fn=lambda:my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)
    
    print ('Training model...')
    print ('RMSE (on training data):')
    training_root_mean_squared_errors = []
    validation_root_mean_squared_errors = []
    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        training_predictions = predictions_to_numpy_array(training_predictions)
        validation_predictions = predictions_to_numpy_array(validation_predictions)
        training_rmse = calculate_root_mean_squared_error(training_predictions, training_targets)
        validation_rmse = calculate_root_mean_squared_error(validation_predictions, validation_targets)
        training_root_mean_squared_errors.append(training_rmse)
        validation_root_mean_squared_errors.append(validation_rmse)
        print("Root Mean squared error for period {} training set: {:.3f} validation set: {:.3f}".format(period, training_rmse, validation_rmse))
        
    print("Model training finished.")
    
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('RMSE per Periods')
    plt.tight_layout()
    plt.plot(training_root_mean_squared_errors, label="training")
    plt.plot(validation_root_mean_squared_errors, label="validation")
    plt.show()
    return linear_regressor

    
    '''calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())
    
    plt.subplot(1,3,3)
    plt.title('Targets vs Predictions')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.scatter(calibration_data['predictions'], calibration_data['targets'])
    plt.show()
    
    print('Final RMSE: {:.3f}'.format(root_mean_squared_error))'''

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

linear_regressor = train_model(learning_rate = 0.00002, steps=600, batch_size=10, training_examples=training_examples, training_targets=training_targets, validation_examples=validation_examples, validation_targets=validation_targets)

california_housing_test_data = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")

test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)
predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets, shuffle=False, num_epochs=1)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = predictions_to_numpy_array(test_predictions)
test_rmse = calculate_root_mean_squared_error(test_predictions, test_targets)
print("Root Mean squared error for the test set: {:.3f} ".format(test_rmse))


'''plt.figure(figsize=(13, 8))
ax = plt.subplot(1,2,1)
ax.set_title('Validation Data')
ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])
plt.scatter(validation_examples['longitude'], validation_examples['latitude'], cmap='coolwarm', c=validation_targets['median_house_value']/validation_targets['median_house_value'].max())

ax = plt.subplot(1,2,2)
ax.set_title('Training Data')
ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])
plt.scatter(training_examples['longitude'], training_examples['latitude'], cmap='coolwarm', c=training_targets['median_house_value']/training_targets['median_house_value'].max())
plt.show()  '''
