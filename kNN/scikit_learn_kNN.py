from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from .data_simple_one import X_train, y_train, x
from .data_digits import digits

X = digits.data
y = digits.target

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

kNN_classifier = KNeighborsClassifier(n_neighbors=3)
kNN_classifier.fit(X_train, y_train)

# 将要预测的数据放入一个矩阵
# X_predict = x.reshape(1, -1)
# y_predict = kNN_classifier.predict(X_predict)
# y_predict[0]

# y_predict = kNN_classifier.predict(X_test)
result = kNN_classifier.score(X_test, y_test)