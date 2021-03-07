# Description：
# Author：朱勇
# Time：2021/3/7 10:05

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.read_csv("task1_data1.csv")
data_result = pd.read_csv("task1_data2.csv")
#
x_labeled = data.iloc[0,:]
#
x = data.drop(['y'],axis=1)
y = data_result.loc[:,'y']
fig1 = plt.figure()
plt.scatter(data.loc[:,'x1'],data.loc[:,'x2'],label='unlabeled')
plt.scatter(x_labeled['x1'],x_labeled['x2'],label='labeled')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('unlabeled data')
plt.legend(loc='upper left')
plt.show()
#
KM = KMeans(n_clusters=2,init='random',random_state=0)
#
KM.fit(x)
#
centers = KM.cluster_centers_
print(centers)
#
y_predict = KM.predict(x)
print(pd.value_counts(y))
print(pd.value_counts(y_predict))
print(accuracy_score(y_predict,y))
#
print(x_labeled,y_predict[0])
y_corrected = []
for i in y_predict:
    if i == 0:
        y_corrected.append(1)
    elif i == 1:
        y_corrected.append(0)
print(y_corrected)
print(pd.value_counts(y_corrected))
y_corrected = np.array(y_corrected)


