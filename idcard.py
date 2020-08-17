import tensorflow as tf
import numpy as np
import random
import os
import sys


random.seed()

trainResultPath="save/idcard2"
ifRestart = False
predictData = None
roundT = 5

argt = sys.argv[1:]
print("argt:%s" % argt)

for v in argt:
    if v == "-restart":
        ifRestart = True
    if v.startswith("-predict="):
        predictData = np.fromstring(v[9:],dtype=np.float32, sep=",")
    if v.startswith("-round="):
        roundT = int(v[7:])

rowCount = 5001

xDataRandom = np.full((rowCount, 4),5,dtype=np.float32)
yTrainDataRandom = np.full((rowCount,2),0,dtype=np.float32)

for i in range(rowCount):
    for j in range(4):
        xDataRandom[i][j] = np.floor( random.random() * 10)

    if xDataRandom[i][2] % 2 == 0:
        yTrainDataRandom[i][0] = 0
        yTrainDataRandom[i][1] =1
    else:
        yTrainDataRandom[i][0] = 1
        yTrainDataRandom[i][1] = 0

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(64,input_dim=64,activation="tanh",use_bias=True))

model.add(tf.keras.layers.Dense(2,input_dim=64,activation="softmax",use_bias=True))

model.compile(loss="binary_crossentropy",optimizer="RMSProp",metrics=['accuracy'])

