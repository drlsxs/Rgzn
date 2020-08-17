import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x1=tf.placeholder(dtype=tf.float32)
x2=tf.placeholder(dtype=tf.float32)
x3=tf.placeholder(dtype=tf.float32)
yT=tf.placeholder(dtype=tf.float32)
w1=tf.Variable(0.1, dtype=tf.float32)
w2=tf.Variable(0.1, dtype=tf.float32)
w3=tf.Variable(0.1, dtype=tf.float32)

n1=x1*w1
n2=x2*w2
n3=x3*w3

y=n1+n2+n3

print(y)

loss=tf.abs(y - yT)
optimizer=tf.train.RMSPropOptimizer(0.001)
train = optimizer.minimize(loss)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
result = sess.run([train,w1,w2,w3,y,loss],feed_dict={x1:90,x2:80,x3:70,yT:85})
print(result)