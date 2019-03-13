import tensorflow as tf

nb_classes = 7

Y = [0, 1, 2, 4, 5, 6, 2, 4]

Y_one_hot = tf.one_hot(Y, nb_classes)

sess = tf.Session()

print(sess.run(Y_one_hot))
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print(sess.run(Y_one_hot))
