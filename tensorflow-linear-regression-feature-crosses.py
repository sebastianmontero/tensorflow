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

def construct_feature_columns(training_examples):
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
    
    bucketized_households = tf.feature_column.bucketized_column(households, get_quantile_based_boundry(training_examples['households'], 7))
    bucketized_longitude = tf.feature_column.bucketized_column(longitude, get_quantile_based_boundry(training_examples['longitude'], 10))
    bucketized_latitude = tf.feature_column.bucketized_column(latitude, get_quantile_based_boundry(training_examples['latitude'], 10))
    bucketized_housing_median_age = tf.feature_column.bucketized_column(housing_median_age, get_quantile_based_boundry(training_examples['housing_median_age'], 7))
    bucketized_median_income = tf.feature_column.bucketized_column(median_income, get_quantile_based_boundry(training_examples['median_income'], 7))
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(rooms_per_person, get_quantile_based_boundry(training_examples['rooms_per_person'], 5))
    return set([bucketized_households, bucketized_longitude, bucketized_latitude, bucketized_housing_median_age, bucketized_median_income, bucketized_rooms_per_person])

def predictions_to_numpy_array(predictions):
    return np.array([item['predictions'][0] for item in predictions])

def calculate_root_mean_squared_error(predictions, targets):
        return math.sqrt(metrics.mean_squared_error(predictions, targets))

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps/periods
       
    my_label = 'median_house_value'
   
    feature_columns = construct_feature_columns(training_examples)
    
    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
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

def bin_latitude (latitude, min, bins):
    bins = [0] * bins;
    bins[math.floor(latitude) - min] = 1
    return bins

def select_and_transform_features(source_df):
    selected_examples = pd.DataFrame()
    selected_examples['median_income'] = source_df['median_income']
    for r in LATITUDE_RANGES:
        selected_examples['lat_range_{}_to_{}'.format(r[0], r[1])] = source_df['latitude'].apply(lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
    return selected_examples
    
def get_quantile_based_boundry(feature_values, num_buckets):
    boundries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundries)
    return [quantiles[q] for q in quantiles.keys()]
    

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

print(training_examples.describe())

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

correlation_dataframe = training_examples.copy()
correlation_dataframe['target'] = training_targets['median_house_value']
correlation_matrix = correlation_dataframe.corr();

lat_start = math.floor(california_housing_dataframe['latitude'].min())
lat_end = math.ceil(california_housing_dataframe['latitude'].max())
'''LATITUDE_RANGES = [[32,33],[33,34]]
LATITUDE_RANGES1 = zip(range(32, 33), range(33, 34))
print(LATITUDE_RANGES)
print(LATITUDE_RANGES1)'''
LATITUDE_RANGES = list(zip(range(lat_start, lat_end - 1), range(lat_start + 1, lat_end)))
#LATITUDE_RANGES = [[32,33]]

#selected_training_examples = select_and_transform_features(training_examples)
#selected_validation_examples = select_and_transform_features(validation_examples)


#plt.hist(training_examples['latitude'], 10)
#plt.show();



linear_regressor = train_model(learning_rate = 1.0, steps=500, batch_size=10, training_examples=training_examples, training_targets=training_targets, validation_examples=validation_examples, validation_targets=validation_targets)

california_housing_test_data = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")

test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)
#selected_test_examples = select_and_transform_features(test_examples)
predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets, shuffle=False, num_epochs=1)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = predictions_to_numpy_array(test_predictions)
test_rmse = calculate_root_mean_squared_error(test_predictions, test_targets)
print("Root Mean squared error for the test set: {:.3f} ".format(test_rmse))
