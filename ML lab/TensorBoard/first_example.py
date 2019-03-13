# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

with tf.name_scope('layer') as scope:
    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

    w_hist = tf.summary.histogram('weight', W)
    b_hist = tf.summary.histogram('bias', b)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1-hypothesis))
    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('train') as scope:
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:/Users/AterCatus/Desktop/ML lab/TensorBoard/logs')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in tqdm_notebook(range(10001)):
        summary, _ = sess.run([merged_summary, train], feed_dict = {X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print('\mHypothesis: ', h, '\nCorrect: ', c, '\nAccuracy: ', a)
