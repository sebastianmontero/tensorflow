import os
from IPython.core.pylabtools import figsize
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
    processed_features['total_rooms'] = clip(processed_features['total_rooms'],0, 9000)
    processed_features['total_bedrooms'] = clip(processed_features['total_bedrooms'],0, 2000)
    processed_features['rooms_per_person'] = processed_features['total_rooms'] / processed_features['population']
    processed_features['rooms_per_person'] = clip(processed_features['rooms_per_person'],0, 6)
    return processed_features

def normalize2(examples, z_score_params):
    normalized_features = pd.DataFrame()
    
    normalized_features['latitude'] = z_score_normalize(examples['latitude'], 'latitude', z_score_params)
    normalized_features['longitude'] = z_score_normalize(examples['longitude'], 'longitude', z_score_params)
    normalized_features['housing_median_age'] = z_score_normalize(examples['housing_median_age'], 'housing_median_age', z_score_params)
    normalized_features['total_rooms'] = z_score_normalize(examples['total_rooms'], 'total_rooms', z_score_params)
    normalized_features['total_bedrooms'] = z_score_normalize(examples['total_bedrooms'], 'total_bedrooms', z_score_params)
    normalized_features['households'] = z_score_normalize(examples['households'], 'households', z_score_params)
    normalized_features['median_income'] = z_score_normalize(examples['median_income'], 'median_income', z_score_params)
    normalized_features['population'] = log_normalize(examples['population'])
    
    return normalized_features

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

def train_model(optimizer, steps, batch_size, hidden_units, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps/periods
       
    my_label = 'median_house_value'
   
    feature_columns = construct_feature_columns(training_examples)
    
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    dnn_regressor = tf.estimator.DNNRegressor(feature_columns = feature_columns, hidden_units=hidden_units, optimizer=optimizer)
    
    training_input_fn=lambda:my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)
    
    print ('Training model...')
    print ('RMSE (on training data):')
    training_root_mean_squared_errors = []
    validation_root_mean_squared_errors = []
    for period in range(0, periods):
        dnn_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
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
    return dnn_regressor

    
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
