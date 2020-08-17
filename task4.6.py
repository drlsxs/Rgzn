import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder(shape=[3],dtype=tf.float32)
yTrain=tf.placeholder(shape=[],dtype=tf.float32)

w=tf.Variable(tf.zeros([3]),dtype=tf.float32)
wn=tf.nn.softmax(w)

n=wn*x
y=tf.reduce_sum(n)
loss=tf.abs(y - yTrain)
optimizer=tf.train.RMSPropOptimizer(0.1)
train=optimizer.minimize(loss)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(2):
    result=sess.run([train,x,w,wn,y,yTrain,loss],feed_dict={x:[90,80,70],yTrain:85})
    print(result[3])
    result = sess.run([train, x, w, wn, y, yTrain, loss], feed_dict={x: [98,95,87], yTrain: 96})
    print(result[3])