import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp

def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))

def train(w, X, y) :
    x_row, x_col = X.shape
    for i in range (x_row):
        x_ext = np.append([1], X[i,:], 0)
        a = w.dot(x_ext)
        y_hat = sigmoid(a) # prediction
        print('i = ', i) 
        print('  w before updating :',np.around(w,decimals=3))
        w = w + alpha * (y[i] - y_hat) * y_hat * (1-y_hat) * x_ext # update 
        print('  w after updating  :',np.around(w,decimals=3))
    return w

def predict(w, data_test, thres):
    data_row, data_col = data_test.shape
    pred = np.empty(((data_row, 1)))
    for i in range (data_row):
        x_ext = np.append([1], X[i,:], 0)
        a = w.dot(x_ext)
        y_hat = sigmoid(a) # prediction
        pred[i] = y_hat
    return pred

def accuracy(y, pred, thres):
    true_positives = 0
    true_negatives = 0
    y_row = len(y)
    for i in range(y_row):
        if y[i] == 1 and pred[i] >= thres:
            true_positives += 1
        elif y[0] == 0 and pred[i] < thres:
            true_negatives += 1
    acc_rate = (true_positives + true_negatives) / y_row
    print(f'Accuracy: {acc_rate}')
    return acc_rate

df = pd.read_csv('iris_sub_dataset.csv')
data = df.values
row, col = data.shape
X = data[:,0:col-1]
y = data[:,col-1]

print('\nLoaded Pandas dataframe\n----------\n',df.head())
print('\nExtraxt data (first 5) \n----------\n',data[0:col])
print('\nX (first 5) \n----------\n',X[0:5])
print('\ny \n----------\n', y)

#a = w0 + w1*xi + w2 * xi
alpha = 0.3
w0 = 0; w1 = 0; w2 = 0; # initial coefficient
w = np.array([w0, w1, w2])
    
thres = 0.5
epochs = 10
acc = np.empty(((epochs, 1)))
x_axis = np.arange(10) + 1
for epoch in range(epochs):
    w = train(w, X, y) # train the model
    print('\n----------\nEpoch = ', epoch)
    print('model w:', np.around(w,decimals=3))
    pred = predict(w, X, thres)
    print('prediction result\n', np.around(pred,decimals=3))
    acc[epoch] = accuracy(y, pred, thres)

plt.plot(x_axis, acc, linestyle='-')  # solid
plt.xlabel("Epoch")
plt.ylabel("Accuracy");
plt.show()


    
