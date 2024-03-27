import numpy as np
from matplotlib import pyplot as plt

data = np.array([[-3,5],[3,1 ],[0,2],[2, 5],[3, 6],[6, 1]])
X = np.c_[data,np.ones(len(data))]
y = np.array([-1,-1,-1,1,1,1])

def svm_sgd_plot(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000
    errors = []
    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)

    errorPerTenThousand = []
    x_epoch = []
    step = 10000
    for i in range(10):
        start = i*step
        end = (i+1)*step
        s1 = errors[start:end]
        rate = np.sum(s1) / step
        errorPerTenThousand.append(rate)
        x_epoch.append(end)

    plt.plot(x_epoch, errorPerTenThousand)
    plt.xlabel('Epoch'); plt.ylabel('Misclassification Rate')
    plt.show()
    print(w)
    return w

w = svm_sgd_plot(X,y)
for d, sample in enumerate(X):
    if d < 3:
        plt.scatter(sample[0], sample[1], s=100, color='b')
    else:
        plt.scatter(sample[0], sample[1], s=100, color='r')
x = np.arange(-3, 6, 0.1)
y = -(w[0]*x + w[2])/w[1]
plt.plot(x, y, color='m')
plt.show()

