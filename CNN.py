import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

raw_data_X = [[3.393, 2.331],
             [3.110, 1.782],
             [1.343, 3.368],
             [3.582, 4.679],
             [2.280, 2.967],
             [7.423, 4.697],
             [5.745, 3.534],
             [9.172, 2.511],
             [7.792, 3.424],
             [7.940, 0.792]]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

x = np.array([8.094, 3.366])

def kNN_classify(k, X_train, y_train, x):
  distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train)]
  nestindexs = np.argsort(distances)

  topK_y = [y_train[i] for i in nestindexs[:k]]
  votes = Counter(topK_y)

  return votes.most_common(1)[0][0]