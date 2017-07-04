import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('../input/train.csv')
y = np.array(train.pop('label'))
x = np.array(train)/255.
plt.imshow(x[10].reshape(28,28), cmap='Greys', interpolation='nearest')

mlp = MLPClassifier(solver='sgd', activation='relu',
                    hidden_layer_sizes=(100,30),
                    learning_rate_init=0.3, learning_rate='adaptive', alpha=0.1,
                    momentum=0.9, nesterovs_momentum=True,
                    tol=1e-4, max_iter=200,
                    shuffle=True, batch_size=300,
                    early_stopping = False, validation_fraction = 0.15,
                    verbose=True)
mlp.fit(x, y)

test = pd.read_csv('../input/test.csv')
x = np.array(test)/225.
y_test = mlp.predict(x)

ImageId = np.arange(1,len(y_test)+1)
submission = pd.DataFrame(data={'ImageId':ImageId, 'Label':y_test})
submission.to_csv('submission-mlp.csv', index=False)
