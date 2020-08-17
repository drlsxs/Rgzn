import tensorflow as tf
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


random.seed()

x = tf.placeholder(tf.float32)
yTrain = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([4, 32], mean=0.5, stddev=0.1), dtype=tf.float32)
b1 = tf.Variable(0, dtype=tf.float32)

xr = tf.reshape(x, [1, 4])

n1 = tf.nn.tanh(tf.matmul(xr, w1)  + b1)

w2 = tf.Variable(tf.random_normal([32, 32], mean=0.5, stddev=0.1), dtype=tf.float32)
b2 = tf.Variable(0, dtype=tf.float32)

n2 = tf.nn.sigmoid(tf.matmul(n1, w2) + b2)

w3 = tf.Variable(tf.random_normal([32, 2], mean=0.5, stddev=0.1), dtype=tf.float32)
b3 = tf.Variable(0, dtype=tf.float32)

n3 = tf.matmul(n2, w3) + b3

y = tf.nn.softmax(tf.reshape(n3, [2]))

loss = tf.reduce_mean(tf.square(y - yTrain))

optimizer = tf.train.RMSPropOptimizer(0.01)

train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

lossSum = 0.0

for i in range(500):

    xDataRandom = [int(random.random() * 10), int(random.random() * 10), int(random.random() * 10), int(random.random() * 10)]
    if xDataRandom[2] % 2 == 0:
        yTrainDataRandom = [0, 1]
    else:
        yTrainDataRandom = [1, 0]

    result = sess.run([train, x, yTrain, y, loss], feed_dict={x: xDataRandom, yTrain: yTrainDataRandom})

    lossSum = lossSum + float(result[len(result) - 1])

    print("i: %d, loss: %10.10f, avgLoss: %10.10f" % (i, float(result[len(result) - 1]), lossSum / (i + 1)))

    #多加了个隐藏层n3,W1定义成【4,32】，W2【32,32】，W3【32,2】