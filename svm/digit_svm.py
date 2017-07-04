import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

COMPONENT_NUM = 0.8

print('Reading Data...')
digit_train = pd.read_csv("../input/train.csv")
digit_train = digit_train.astype(np.float64)

train_set = digit_train.drop("label", axis = 1)
train_set = np.array(train_set)/255.
train_target = digit_train.label

digit_test = pd.read_csv("../input/test.csv")
digit_test = digit_test.astype(np.float64)
test_set = np.array(digit_test)/255.

print('PCA Reduction...')
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_set)
train_set = pca.transform(train_set)
test_set = pca.transform(test_set)

print('SVM Training...')
classifier = svm.SVC(C=10)
classifier.fit(train_set, train_target)

predicted = classifier.predict(test_set)
predicted = predicted.astype(int)

print('Submitting...')
submission = pd.DataFrame({"ImageId": digit_test.index.values+1, "Label": predicted})
submission.to_csv("Digit_svm.csv", index=False)
