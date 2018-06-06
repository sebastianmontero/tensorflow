from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import tensorflow as tf
import numpy as np

def plot_object(points):
    tri = Delaunay(points).convex_hull
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=tri, shade=True, cmap=cm.Blues, lw=0.5)
    ax.set_xlim3d(-5,5)
    ax.set_ylim3d(-5,5)
    ax.set_zlim3d(-5,5)
    plt.show()
    
def create_cube(bottom_lower=(0,0,0), side_length=5):
    bottom_lower = np.array(bottom_lower)
    points = np.vstack([bottom_lower,
                       bottom_lower + [0,side_length,0],
                       bottom_lower + [side_length,side_length,0],
                       bottom_lower + [side_length, 0, 0],
                       bottom_lower + [0,0,side_length],
                       bottom_lower + [0,side_length, side_length],
                       bottom_lower + [side_length,side_length,side_length],
                       bottom_lower + [side_length,0,side_length],
                       bottom_lower])
    return points

def translate(points, amount):
    return tf.add(points, amount)

def rotate_around_z(points, theta):
    theta = float(theta)
    rotation_matrix = tf.stack([[tf.cos(theta), tf.sin(theta), 0],
                                [-tf.sin(theta), tf.cos(theta), 0],
                                [0, 0, 1]])
    return tf.matmul(tf.to_float(points), tf.to_float(rotation_matrix))

cube_1 = create_cube(side_length = 2)

points = tf.constant(cube_1, dtype=tf.float32)
translation_amount = tf.constant([3,-3,0], dtype=tf.float32)

translation_op = translate(points, translation_amount)
rotate_op = rotate_around_z(translation_op, 75)

with tf.Session() as session:
   cube_3 = session.run(rotate_op)

plot_object(cube_3)
