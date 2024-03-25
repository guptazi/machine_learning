import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('diamonds.csv')
X=df[['carat','depth','table','x','y','z']]
df['cut']=pd.Categorical(df['cut'])
df['code']=df['cut'].cat.codes
Y=df['code']
print(df)
x_train,x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
model=LinearSVC()
ovo=OneVsOneClassifier(model)
ovo.fit(x_train,y_train)

data_to_predict=pd.DataFrame({'carat':[0.75],'depth':[62.2],'table':[55],'x':[5.83],'y':[5.87],'z':[3.64]})
predict=ovo.predict(data_to_predict)
prediction_to_code = dict(enumerate(df['cut'].cat.categories))
predicted_categories = [prediction_to_code[code] for code in predict]
print(predicted_categories)
