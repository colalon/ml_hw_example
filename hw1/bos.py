from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from my_LR import my_LR
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()
target = boston.target
target = target.reshape((506,1))
data1 = boston.data
data = boston.data.copy()
'''
for i in range (13):
    data[:,i] = (data1[:,i]-data1[:,i].mean())/data1.std()
    '''

rand = np.random.permutation(data.shape[0])
data = data[rand]
target = target[rand]

x_train = data[0:350]
x_test = data[350:]
y_train = target[0:350]
y_test = target[350:]

l=my_LR()
l.fit(x_train,y_train)
a=l.get_Params()
yt=l.predict(x_test)
print ('score',l.score(x_train,y_train))
plt.scatter(yt,y_test)
t = LinearRegression()

#t.fit(x_train,y_train)

t.fit(x_train,y_train)
aaa=t.predict(x_test)
score=t.score(x_test,y_test)

t_weights = np.zeros(x_train.shape[1])
t_bias = 0
t_bias=t.predict(np.zeros((1,13)))
for i in range (x_train.shape[1]):
    w = np.zeros((1,13))
    w[0,i]=1.0
    t_weights[i] = t.predict(w) - t_bias
    
plt.scatter(aaa,y_test)


plt.figure()
plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(222)
plt.plot([0, 1], [0, 2])

plt.subplot(223)
plt.plot([0, 1], [0, 3])

plt.subplot(224)
plt.plot([0, 1], [0, 4])
plt.show()