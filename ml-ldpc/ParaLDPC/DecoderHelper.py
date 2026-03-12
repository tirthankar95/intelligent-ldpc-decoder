import numpy as np
import math
import Global
import random
import matplotlib.pyplot as plt
import tensorflow as tf

def ATANH(L):
    (r,c)=L.shape
    for i in range(r):
        for j in range(c):
            if L[i,j]!=np.nan:
                try:
                    L[i,j]=math.atanh(L[i,j])
                except:
                    L[i,j]=math.atanh(np.sign(L[i,j])*0.99)
    return L
#My approximation of tanh using ML methods.
def tanh_mine(x):
    x[np.where(np.isnan(x)==True)]=-9999
    indx1 =np.where(( (x!=-9999) & (x>=-1.2) & (x<1.2)))
    indx2 = np.where(((x!=-9999) & (x >= 1.2)))
    indx3 = np.where(((x!=-9999) & (x < -1.2)))
    x[indx1]=0.8435*x[indx1]+0.0003
    x[indx2]=1
    x[indx3]=-1
    x[np.where(x==-9999)] = np.nan
    return x
#My approximation of atanh using ML methods.
def atanh_mine(x):
    x[np.where(np.isnan(x) == True)] = -9999
    indx1=np.where(( (x!=-9999) & (x<-0.85) ))
    indx2=np.where(( (x!=-9999) & (x>=-0.85) & (x<-0.75) ))
    indx3=np.where(( (x!=-9999) & (x>=-0.75) & (x< 0.75) ))
    indx4=np.where(( (x!=-9999) & (x>= 0.75) & (x< 0.85) ))
    indx5=np.where(( (x!=-9999) & (x>=0.85) ))
    x[indx1] = 6.944 * x[indx1] + 4.712
    x[indx2] = 1.777 * x[indx2] + 0.330
    x[indx3] = 1.231 * x[indx3] + 0.003
    x[indx4] = 1.815 * x[indx4] - 0.334
    x[indx5] = 8.347 * x[indx5] - 5.884
    x[np.where(x == -9999)] = np.nan
    return x
def genAll(m):
    Global.msg=[];Global.example_cnt=0
    num=min(2**Global.k,m)
    Global.msg=np.random.randint(low=0,high=2,size=(num,Global.k)).tolist()
# General helper functions.
def demod(approx):
    res=approx
    indx=np.where(res>=0)
    indx1=np.where(res<0)
    res[indx]=0
    res[indx1]=1
    return res
def MIN(L):
    (r,c)=L.shape
    L[np.isnan(L)==True]=999
    for i in range(r):
        minF=minS=999
        for j in range(c):
            if minF>L[i,j]:
                minS=minF
                minF=L[i,j]
            elif minS>L[i,j]:
                minS=L[i,j]
        for j in range(c):
            if L[i,j]!=minF and L[i,j]!=999:
                L[i,j]=minF
            elif L[i,j]!=999:
                L[i,j]=minS
    L[L==999]=np.nan
    return L
#My approximation of atanh_mine_poly using ML methods.
def atanh_mine_poly(x):
    return 3.91*(x**5)-2.72*(x**3)+1.48*x
#My approximation of atanh_mine_poly using ML methods.
N = 3  # 2n+1 -dimensional polynomial.
model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='linear'),
        tf.keras.layers.Dense(2, activation='linear'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
model.compile(optimizer='sgd',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['MeanSquaredError'])
def atanh_NNtrain():
    lb = -0.9999;RANGE = 100000
    offset = -2 * lb / RANGE
    X = np.array([lb + i * offset for i in range(RANGE+1)])
    for i in range(RANGE):
        index = random.randint(0, i)
        tmp = X[index]
        X[index] = X[i]
        X[i] = tmp
    Y = np.arctanh(X)
    Xtest = X[0:int(0.1 * RANGE)].reshape(1, int(0.1 * RANGE))
    Ytest = Y[0:int(0.1 * RANGE)].reshape(1, int(0.1 * RANGE))
    Xtr = X[int(0.1 * RANGE):RANGE].reshape(1, int(0.9 * RANGE))
    Ytr = Y[int(0.1 * RANGE):RANGE].reshape(1, int(0.9 * RANGE))
    Xtr_M6 = np.zeros((N, Xtr.shape[1]))
    Xtest_M6 = np.zeros((N, Xtest.shape[1]))
    Xtr_M6[0] = Xtr[0]
    Xtest_M6[0] = Xtest[0]
    power=1
    for i in range(N - 1):
        power += 2
        Xtr_M6[i + 1] = (Xtr_M6[i] ** power)
        Xtest_M6[i + 1] = (Xtest_M6[i] ** power)
    model.fit(Xtr_M6.T,Ytr.T,epochs=5)
    print(model.evaluate(Xtest_M6.T,Ytest.T,verbose=2))
def atanh_mine_NN(x):
    xc=x.flatten()
    xtest=np.zeros((N,xc.shape[0]))
    xtest[0]=xc
    power=1
    for i in range(N - 1):
        power += 2
        xtest[i + 1] = (xtest[i] ** power)
    return model.predict(np.array(xtest.T)).reshape(x.shape)
def test():
    atanh_NNtrain()
    lb = -0.9999;RANGE = 100
    offset = -2 * lb / RANGE
    X=np.array([lb + i * offset for i in range(RANGE+1)])
    X=X.reshape(1,X.shape[0])
    Y=np.arctanh(X)
    YP_NN=atanh_mine_NN(X)
    YP_Poly=atanh_mine_poly(X)
    plt.plot(X,Y,'r.-')
    plt.plot(X,YP_NN,'g.-')
    plt.plot(X,YP_Poly,'b.-')
    plt.legend(('Actual','NN','Poly'))
    plt.show()
#test()