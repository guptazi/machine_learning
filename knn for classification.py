import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#read data
df=pd.read_csv('Prostate_cancer-training.csv')
x=df[['radius','texture','perimeter','area','smoothness','compactness','symmetry','fractal_dimension']]
y=df['diagnosis_result']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_7 = KNeighborsClassifier(n_neighbors=7)
knn_9 = KNeighborsClassifier(n_neighbors=9)

knn_5.fit(X_train, y_train)
knn_7.fit(X_train, y_train)
knn_9.fit(X_train, y_train)

predictions_5 = knn_5.predict(X_test)
predictions_7 = knn_7.predict(X_test)
predictions_9 = knn_9.predict(X_test)


print(f'prediction for k=5{predictions_5}')
print(f'prediction for k=7 {predictions_7}')
print(f'prediction for k=9 {predictions_9}')
