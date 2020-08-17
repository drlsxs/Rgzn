import tensorflow as tf
import numpy as np
import random
import os
import sys

ifRestartT = False

predictData = None

argt = sys.argv[1:]

for v in argt:
    if v == "-restart":
        ifRestartT = True
    if v.startswith("-predict="):
        tmpStr = v[len("-predict="):]
        predictData = np.fromstring(tmpStr, dtype=np.float32, sep=",")

print("predictData: %s" % predictData)

trainResultPath = "./save/idcard2"

random.seed()

x = tf.placeholder(tf.float32)
yTrain = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([4, 32], mean=0.5, stddev=0.1), dtype=tf.float32)
b1 = tf.Variable(0, dtype=tf.float32)

xr = tf.reshape(x, [1, 4])

n1 = tf.nn.tanh(tf.matmul(xr, w1) + b1)

w2 = tf.Variable(tf.random_normal([32, 2], mean=0.5, stddev=0.1), dtype=tf.float32)
b2 = tf.Variable(0, dtype=tf.float32)

n2 = tf.matmul(n1, w2) + b2

y = tf.nn.softmax(tf.reshape(n2, [2]))

loss = tf.reduce_mean(tf.square(y - yTrain))

optimizer = tf.train.RMSPropOptimizer(0.01)

train = optimizer.minimize(loss)

sess = tf.Session()

if ifRestartT:
    print("force restart...")
    sess.run(tf.global_variables_initializer())
elif os.path.exists(trainResultPath + ".index"):
    print("loading: %s" % trainResultPath)
    tf.train.Saver().restore(sess, save_path=trainResultPath)
else:
    print("train result path not exists: %s" % trainResultPath)
    sess.run(tf.global_variables_initializer())

if predictData is not None:
    result = sess.run([x, y], feed_dict={x: predictData})
    print(result[1])
    print(y.eval(session=sess, feed_dict={x: predictData}))
    sys.exit(0)

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

    if os.path.exists("save.txt"):
        os.remove("save.txt")
        print("saving...")
        tf.train.Saver().save(sess, save_path=trainResultPath)

resultT = input('Would you like to save? (y/n)')

if resultT == "y":
    print("saving...")
    tf.train.Saver().save(sess, save_path=trainResultPath)
