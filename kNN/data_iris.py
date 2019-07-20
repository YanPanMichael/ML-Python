import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target