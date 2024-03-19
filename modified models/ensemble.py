from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix,r2_score,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Assuming X_train_transformed, X_test_transformed, y_train, and y_test are already prepared

data=pd.read_csv('c:\\Users\\sgupta1\\Desktop\\modified model\\updated data.csv',low_memory=False)
data_req_=data[['Adm. Sys.','Gov. Cont.','Func. Class','Terrain','Land Use','Operation','Acc. Ctrl.','No. Lns.','Spd Limit','GPS Latitude','GPS Longitude','AADT','Degree of Curve','Percent of Grade']]
data_req=data_req_.dropna()
x=data_req.drop(columns='AADT')
y=data_req['AADT']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

transf=ColumnTransformer([
    ('onehot',OneHotEncoder(),[0,1,2,3,4,5,6]),
    ('scaler',StandardScaler(),[7,8,9,10,11,12])
])

x_train_transformed=transf.fit_transform(x_train).astype(float)
x_test_transformed=transf.transform(x_test).astype(float)



# Define base models
base_models = [
    ('knn1', KNeighborsRegressor(n_neighbors=5)),
    ('knn2', KNeighborsRegressor(n_neighbors=10)),
    ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('gbm', XGBRegressor(n_estimators=10, random_state=42))
]

# Define the meta-model
meta_model = LinearRegression()

# Create the stacking ensemble
stacking_ensemble = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the stacking ensemble
stacking_ensemble.fit(x_train_transformed, y_train)

# Predictions
ensemble_preds = stacking_ensemble.predict(x_test_transformed)

# Evaluate the ensemble model
mse = mean_squared_error(y_test, ensemble_preds)
print(f"Mean Squared Error of the ensemble model: {mse}")
