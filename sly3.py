import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


wholeData=np.loadtxt(open("data.txt",encoding="cp1252"),delimiter=",",dtype=np.float32,skiprows=0,converters={0:lambda s: 0.0,1:lambda s:(float(s)+1)})
rowCount=int(wholeData.size / wholeData[0].size)

print("%d row found." % rowCount)

goodCount=0