import tensorflow as tf
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.placeholder(dtype=tf.float32)
b = tf.nn.sigmoid(a)

sess = tf.Session()

for i in range(5):
    aData=[int(random.random() * 40 - 20)]
    print(sess.run(b, feed_dict={a:aData}))




