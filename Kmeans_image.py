# Description：
# Author：朱勇
# Time：2021/3/7 10:55

import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from sklearn.cluster import KMeans

#图像导入
img = io.imread('1.jpg')
print(type(img))
print(img.shape)
print(img)
img_width = img.shape[1]
img_height = img.shape[0]
print(img_width,img_height)
#图像转化
img_data = img.reshape(-1,3)
print(img_data.shape)
x = img_data
#模型建立
model = KMeans(n_clusters=3,random_state=0)
model.fit(x)
label = model.predict(x)
print(label)
print(pd.value_counts(label))
label = label.reshape([img_height,img_width])
#转化灰度
label = 1/(label+1)
plt.imshow(label)
plt.show()
io.imsave('result_k3.png',label)