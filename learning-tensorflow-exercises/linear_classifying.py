from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

X_values, y_flat = make_blobs(n_samples=800, n_features=2, centers=3, random_state=500)

y_flat_reshaped = y_flat.reshape(-1,1)

y = OneHotEncoder().fit_transform(y_flat_reshaped).todense()
y=np.array(y)

#plt.scatter(X_values[:,0], X_values[:,1], c=y_flat, alpha=0.4, s=150)
#plt.show()

X_train, X_test, y_train, y_test, y_train_flat, y_test_flat = train_test_split(X_values, y, y_flat)

n_features = X_values.shape[1]
n_classes = len(set(y_flat))

weights_shape = (n_features, n_classes)

W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))
X = tf.placeholder(dtype=tf.float32)
Y_true = tf.placeholder(dtype=tf.float32)

bias_shape = (1, n_classes)
b = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(bias_shape))

Y_pred = tf.matmul(X,W) + b

loss_function = tf.losses.softmax_cross_entropy(Y_true, Y_pred)

learner = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(5000):
        result = session.run(learner, {X:X_train, Y_true: y_train})
        if i % 100 == 0:
            print('Error: {:.6f}'.format(session.run(loss_function, {X:X_test, Y_true:y_test})))
    
    y_pred = session.run(Y_pred, {X:X_test})
    W_final, b_final = session.run([W, b])
        
predicted_y_values = np.argmax(y_pred, axis=1)

h = 1
x_min, x_max = X_values[:, 0].min() - 2 * h, X_values[:, 0].max() + 2 * h
y_min, y_max = X_values[:, 1].min() - 2 * h, X_values[:, 1].max() + 2 * h
x_0, x_1 = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
decision_points = np.c_[x_0.ravel(), x_1.ravel()]

# We recreate our model in NumPy
Z = np.argmax(decision_points @ W_final[[0,1]] + b_final, axis=1)

# Create a contour plot of the x_0 and x_1 values
Z = Z.reshape(x_0.shape)
plt.contourf(x_0, x_1, Z, alpha=0.1)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train_flat, alpha=0.3)
plt.scatter(X_test[:,0], X_test[:,1], c=predicted_y_values, marker='x', s=200)

plt.xlim(x_0.min(), x_0.max())
plt.ylim(x_1.min(), x_1.max())
plt.show()

#plt.plot(X_test[:,0], X_test[:,1], 'rx', markersize=20)
#plt.show()