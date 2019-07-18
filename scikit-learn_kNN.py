from sklearn.neighbors import KNeighborsClassifier

kNN_classifier = KNeighborsClassifier(n_neighbors=6)

kNN_classifier.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
  metric_params=None, n_jobs=1, n_neighbors=6, p=2,
  weights='uniform')

# 将要预测的数据放入一个矩阵
X_predict = x.reshape(1, -1)
y_predict = kNN_classifier.predict(X_predict)
y_predict[0]