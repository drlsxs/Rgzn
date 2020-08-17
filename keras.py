import tensorflow.contrib.keras as k
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed()

rowSize = 4
rowCount = 8192

xDataRandom = np.full((rowCount, rowSize), 5, dtype=np.float32)
yTrainDataRandom = np.full((rowCount, 2), 0, dtype=np.float32)
for i in range(rowCount):
    for j in range(rowSize):
        xDataRandom[i][j] = np.floor(random.random() * 10)
        if xDataRandom[i][2] % 2 == 0:
            yTrainDataRandom[i][0] = 0
            yTrainDataRandom[i][1] = 1
        else:
            yTrainDataRandom[i][0] = 1
            yTrainDataRandom[i][1] = 0

model = k.models.Sequential()

model.add(k.layers.Dense(32, input_dim=4, activation='tanh'))

model.add(k.layers.Dense(32, input_dim=32, activation='sigmoid'))

model.add(k.layers.Dense(2, input_dim=32, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer="RMSProp", metrics=['accuracy'])

model.fit(xDataRandom, yTrainDataRandom, epochs=1000, batch_size=64, verbose=2)

xTestData = np.array([[4, 5, 3, 7], [2, 1, 2, 6], [9, 8, 7, 6], [0, 1, 9, 3], [3, 3, 0, 3]], dtype=np.float32)

for i in range(len(xTestData)):
    resultAry = model.predict(np.reshape(xTestData[i], (1, 4)))
    print("x: %s, y: %s" % (xTestData[i], resultAry))

