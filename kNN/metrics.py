def accuracy_score(y_ture, y_predict):
  assert y_ture.shape[0] == y_predict.shape[0], \
    "the size of y_ture must be equal to the size of y_predict"
  return sum(y_predict == y_ture) / len(y_ture)