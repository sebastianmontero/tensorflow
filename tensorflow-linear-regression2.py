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


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0

california_housing_dataframe['rooms_per_person'] = california_housing_dataframe['total_rooms'] / california_housing_dataframe['population']

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None): 
    features = {key: np.array(value) for key, value in dict(features).items()}
    temp = (features,targets)
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size = 1000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(learning_rate, steps, batch_size, input_feature='total_rooms'):
    periods = 10
    steps_per_period = steps/periods
       
    my_label = 'median_house_value'
   
    features_data = california_housing_dataframe[[input_feature]]
    
    feature_columns = [tf.feature_column.numeric_column(input_feature)]
    
    targets = california_housing_dataframe[my_label]
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    linear_regressor = tf.estimator.LinearRegressor(feature_columns = feature_columns, optimizer=optimizer)
    
    input_fn=lambda:my_input_fn(features_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(features_data, targets, shuffle=False, num_epochs=1)
    
    plt.figure(figsize=(15,6))
    plt.subplot(1,3,1)
    plt.title('Learned line by period')
    plt.ylabel(my_label)
    plt.xlabel(input_feature)
    sample = california_housing_dataframe.sample(300)
    plt.scatter(sample[input_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    
    print ('Training model...')
    print ('RMSE (on training data):')
    root_mean_squared_errors = []
    for period in range(0, periods):
        linear_regressor.train(input_fn=input_fn, steps=steps_per_period)
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        mean_squared_error = metrics.mean_squared_error(predictions, targets)
        root_mean_squared_error = math.sqrt(mean_squared_error)
        print("Root Mean squared error for period {} {:.3f}".format(period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)
        y_extents = np.array([0, sample[my_label].max()])
        weight = linear_regressor.get_variable_value('linear/linear_model/{}/weights'.format(input_feature))[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[input_feature].max()), sample[input_feature].min())
        
        y_extents = x_extents*weight + bias
        plt.plot(x_extents, y_extents, color=colors[period]) 
        
    print("Model training finished.")
    
    plt.subplot(1,3,2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('RMSE per Periods')
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())
    
    plt.subplot(1,3,3)
    plt.title('Targets vs Predictions')
    plt.xlabel('predictions')
    plt.ylabel('targets')
    plt.scatter(calibration_data['predictions'], calibration_data['targets'])
    plt.show()
    
    print('Final RMSE: {:.3f}'.format(root_mean_squared_error))
    
train_model(learning_rate = 0.05, steps=500, batch_size=10, input_feature='rooms_per_person')