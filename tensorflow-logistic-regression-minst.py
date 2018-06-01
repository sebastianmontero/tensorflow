
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import io
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

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
    return set([tf.feature_column.numeric_column('pixels', shape=784)])


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
    

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps / periods
    
    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)
    predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)
       
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(), n_classes=10, optimizer=optimizer, config = tf.estimator.RunConfig(keep_checkpoint_max=1))
    
    
    print ('Training model...')
    print ('Log Loss (on training data):')
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        
        linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        
        training_predictions = list(linear_classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)
        
        validation_predictions = list(linear_classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)
        
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
        print("Log Loss for period {} training set: {:.3f} validation set: {:.3f}".format(period, training_log_loss, validation_log_loss))
        print("------------------------------------------------------")
        # print("Log Loss2 for period {} validation set: {:.3f}".format(period, validation_log_loss2))
        
    print("Model training finished.")
    
    _=map(os.remove, glob.glob(os.path.join(linear_classifier.model_dir, 'events.out.tfevents*')))
    
    final_predictions = linear_classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print('Final prediction accuracy:  {:.3f}'.format(accuracy))
    
    plt.ylabel('LogLoss')
    plt.xlabel('Periods')
    plt.title('LogLoss per Periods')
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.show()
    
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    print(cm.sum(axis=1))
    print(cm.sum(axis=1)[:, np.newaxis])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap='bone_r')
    ax.set_aspect(1)
    plt.title('Confusion matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return linear_classifier


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

minst_csv = os.path.join(os.path.dirname(__file__), 'data/mnist_train_small.csv')

minst_dataframe = pd.read_csv(io.open(minst_csv, 'r'), sep=",", header=None)
minst_dataframe = minst_dataframe.head(10000)
minst_dataframe = minst_dataframe.reindex(np.random.permutation(minst_dataframe.index))

training_targets, training_examples = parse_labels_and_features(minst_dataframe.head(7500))
validation_targets, validation_examples = parse_labels_and_features(minst_dataframe.tail(2500))

'''rand_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()
ax.matshow(training_examples.loc[rand_example].values.reshape(28,28))
ax.set_title('Label: {}'.format(training_targets.loc[rand_example]))
ax.grid(False)
plt.show()'''

classifier = train_model(learning_rate=0.01, steps=10, batch_size=10, training_examples=training_examples, training_targets=training_targets, validation_examples=validation_examples, validation_targets=validation_targets)
'''training_examples = preprocess_features(california_housing_dataframe.head(12000))
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

