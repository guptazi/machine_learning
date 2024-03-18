#qestion 3
import matplotlib.pyplot as plt
import numpy as np
from csv import reader
with open('homework2-dataset-iris.csv','r',encoding='utf-8-sig') as read_obj:
    csv_reader=reader(read_obj)
    list_data=list(csv_reader)

column_0=[]
column_1=[]
column_2=[]
column_3=[]

for row in list_data[1:]:
    float_row=[]

    for element in row[:1]:
        float_row.append(float(element))
    
    column_0.append(float_row)

for row in list_data[1:]:
    float_row=[]

    for element in row[1:2]:
        float_row.append(float(element))
    
    column_1.append(float_row)


for row in list_data[1:]:
    float_row=[]

    for element in row[2:3]:
        float_row.append(float(element))
    
    column_2.append(float_row)

for row in list_data[1:]:
    float_row=[]

    for element in row[3:4]:
        float_row.append(float(element))
    
    column_3.append(float_row)

plt.figure(figsize=(6,6))

#for 1st graph
plt.scatter(column_0,column_1)
plt.title('Visualization of Iris Dataset')
plt.xlabel('Sepal_length')
plt.ylabel('Sepal_width')
plt.show()


#for 2nd grapg
#plt.scatter(column_2,column_3)
#plt.title('Visualization of Iris Dataset')
#plt.xlabel('petal_length')
#plt.ylabel('petal_width')
#plt.show()

