#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###   A factory has the results two tests of microships. The test shown that some microships 
###   have defects and we would like to use these data to train a machine learning 
###   model to use in oreder to predict the future results. The defects microships 
###   have the y_test 0  while the normal microships have 1 result.  ''''
###    import the libraries used in the investigation


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
####   Import the data set. The test result is from 0 to 1, so we don't need to 
####   scale the dataset

data_0 = pd.read_csv('ex2data2.txt')


X = data_0.iloc[:,[0,1] ].values
y= data_0.iloc[:,2].values

####  create the training set and the test set by splitting the data_0 ''''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#####  the scalling of the data 
#####  from sklearn.preprocessing import StandardScaler
#####  scaling_factor = StandardScaler() 
#####   X_train = scaling_factor.fit_transform(X_train)
#####   X_test = scaling_factor.transform(X_test)   ''''

####create the model 
from sklearn.svm import SVC
### gaussion kernel is rbf, 
model = SVC(kernel = 'rbf', random_state = 0)
### fit the kernel to the training set
model.fit(X_train, y_train)
#### prediction of the result
y_prediction = model.predict(X_test)

### confusion matrix gives the number of correct prediction and incorrect prediction
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_prediction)



#### boundary is defined by the value boundary = 0.025
boundary = 0.025

a_min, a_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1,
b_min, b_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1,
aa, bb = np.meshgrid(np.arange(a_min, a_max, boundary), 
                     np.arange(b_min, b_max, boundary))
function_0 = model.predict(np.c_[aa.ravel(),bb.ravel()])
function_0 = function_0.reshape(aa.shape)
plt.contourf(aa, bb, function_0, cmap = plt.cm.coolwarm, alpha = 0.8)

##### Import the matplotlib for the visualization of the Dataset.

from matplotlib.colors import ListedColormap

for n,m in enumerate(np.unique(y_train)):
        plt.scatter(X_train[y_train == m,0], X_train[y_train == m,1], 
                c = ListedColormap(('orange','black'))(n), label = m )
        
plt.xlabel('Microship Test 1')
plt.ylabel('Microship Test 2')
plt.xlim(aa.min(), aa.max())
plt.xlim(bb.min(), bb.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM')



#### Visualization the  results the test set  of microships

plt.contourf(aa, bb, function_0, cmap = plt.cm.coolwarm, alpha = 0.8)
from matplotlib.colors import ListedColormap
for n,m in enumerate(np.unique(y_train)):
        plt.scatter(X_test[y_test == m,0], X_test[y_test == m,1], 
                c = ListedColormap(('orange','black'))(n), label = m )
        
plt.xlabel('Microship Test 1')
plt.ylabel('Microship Test 2')
plt.xlim(aa.min(), aa.max())
plt.xlim(bb.min(), bb.max())
plt.xticks(())
plt.yticks(())
plt.title('Test-set /SVM')

#####   Bibliography
#####   https://pythonspot.com/
















































