import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x = tf.placeholder(shape=[3], dtype=tf.float32)
yTrain = tf.placeholder(shape=[], dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

y = tf.reduce_sum(tf.nn.sigmoid(x * w + b))

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(500):
    print(sess.run([train, y, yTrain, loss], feed_dict={x: [1, 1, 1], yTrain: 2}))
    print(sess.run([train, y, yTrain, loss], feed_dict={x: [1, 0, 1], yTrain: 1}))
    print(sess.run([train, y, yTrain, loss], feed_dict={x: [1, 2, 3], yTrain: 3}))