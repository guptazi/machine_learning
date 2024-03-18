import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df=pd.read_csv('HW6-Q1-data.csv')

replace1={'C3':3, 'C24':24, 'C8':8, 'C9':9, 'C1':1, 'C15':15, 'C27':27, 'C20':20, 'C4':4, 'C2':2, 'C10':10, 'C17':17, 'C33':33,'C26':26}
df['City_Code']=df['City_Code'].map(replace1)

replace2={'Rented':1,'Owned':2}
df['Accomodation_Type']=df['Accomodation_Type'].map(replace2)

replace3={'Individual':1,'Joint':2}
df['Reco_Insurance_Type']=df['Reco_Insurance_Type'].map(replace3)

replace4={'Yes':1,'No':2}
df['Is_Spouse']=df['Is_Spouse'].map(replace4)

replace5={'X1':1,'X2':2,'X4':4,'X3':3,'X5':5,'X6':6}
df['Health Indicator']=df['Health Indicator'].map(replace5)

data=df.values
row,col=data.shape
data2=data[1:row,0:col]
x=data2.astype(float)
xnorm=StandardScaler().fit_transform(x)

pca=PCA(n_components=2)
pcacomponents=pca.fit_transform(xnorm)
pcadf=pd.DataFrame(data=pcacomponents,columns=['1st PC','2nd PC'])
sns.scatterplot(data=pcadf,x='1st PC',y='2nd PC')
plt.show()
variance = pca.explained_variance_ratio_
print(variance)

#PCA 3 components
pca3=PCA(n_components=3)
pcacomponents=pca3.fit_transform(xnorm)
pcadf=pd.DataFrame(data=pcacomponents,columns=['1st PC','2nd PC','3rd PC'])

fig=plt.figure(figsize=(9,9))
axes=fig.add_subplot(111, projection='3d')  
axes.set_title('PCA Components')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2') 
axes.set_zlabel('PC3')
axes.scatter(pcadf['1st PC'],pcadf['2nd PC'],pcadf['3rd PC'], s=10)
plt.show()

variance3 = pca3.explained_variance_ratio_
print(variance3)
