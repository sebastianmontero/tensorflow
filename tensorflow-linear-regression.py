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

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None): 
    features = {key: np.array(value) for key, value in dict(features).items()}
    temp = (features,targets)
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size = 1000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
    
    
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0

features_data = california_housing_dataframe[['total_rooms']]

feature_columns = [tf.feature_column.numeric_column('total_rooms')]

targets = california_housing_dataframe['median_house_value']

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(feature_columns = feature_columns, optimizer=optimizer)

linear_regressor.train(input_fn=lambda:my_input_fn(features_data, targets), steps=500)

prediction_input_fn = lambda: my_input_fn(features_data, targets, shuffle=False, num_epochs=1)

predictions = linear_regressor.predict(input_fn=prediction_input_fn)

predictions = np.array([item['predictions'][0] for item in predictions]) 
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

min_median_house_value = california_housing_dataframe['median_house_value'].min()
max_median_house_value = california_housing_dataframe['median_house_value'].max()
min_max_diff = max_median_house_value - min_median_house_value

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(predictions)
print(calibration_data.describe())

print("Min Median house value: {:.2f}".format(min_median_house_value))
print("Max Median house value: {:.2f}".format(max_median_house_value))
print("Diff Median house value: {:.2f}".format(min_max_diff))
print("Mean squared error(on training data): {:.2f}".format(mean_squared_error))
print("Mean squared error(on training data): {:.2f}".format(mean_squared_error))
print("Root Mean squared error(on training data): {:.3f}".format(root_mean_squared_error))

sample = california_housing_dataframe.sample(300)

x_0 = sample['total_rooms'].min()
x_1 = sample['total_rooms'].max()

weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = x_0*weight + bias;
y_1 = x_1*weight + bias;

plt.plot([x_0, x_1],[y_0,y_1], c='r')
plt.xlabel('total_rooms')
plt.ylabel('median_house_value')

plt.scatter(sample['total_rooms'], sample['median_house_value'])

plt.show()


#print(california_housing_dataframe)
#print(california_housing_dataframe.describe())



    