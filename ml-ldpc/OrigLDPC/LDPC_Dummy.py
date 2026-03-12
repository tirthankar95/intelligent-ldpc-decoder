#!/usr/bin/env python
# coding: utf-8

# In[202]:


import numpy as np
import math
MX_MODEL=5
models=[0,1,2,4]
n=6
k=4
H=np.array([[0, 1, 0, 1, 0, 1],[1, 0, 1, 0, 1, 0]])


# In[203]:


def demod(approx):
    res=approx
    indx=np.where(res>=0)
    indx1=np.where(res<0)
    res[indx]=0
    res[indx1]=1
    return res

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


# In[204]:


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


# In[205]:


def decode(H,c_Rx):
    mx_iter=50
    l_intrinsic = np.multiply(2 / std**2, c_Rx)
    d_bits=[[] for i in range(MX_MODEL)]
    for model in models:
        lin=l_intrinsic
        indx0=np.where(H==0)
        L=H.astype(np.float)
        L[indx0]=np.nan
        L=np.multiply(L,lin)
        if model==4:# Defaut MODEL
            d_bits[model] = demod(l_intrinsic)
            continue
        if model==0:# Origianl MODEL
            for i in range(mx_iter):
                L=np.tanh(L/2)
                L_=np.nanprod(L,axis=1).reshape(n-k,1)
                L=np.divide(L_,L)
                L=2*ATANH(L)
                lin=(lin+np.nansum(L,axis=0)).reshape(1,n)
                L=lin-L
            d_bits[model]=demod(lin)
        if model==1:# Bad MODEL
            for i in range(mx_iter):
                L=np.tanh(L/2)
                L_=np.nanprod(L,axis=1).reshape(n-k,1)
                L=np.divide(L_,L)
                lin=(lin+np.nansum(L,axis=0)).reshape(1,n)
                L=lin-L
            d_bits[model]=demod(lin)
        if model==2:# MinSum MODEL
            for i in range(mx_iter):
                S=np.sign(L)
                S=np.nanprod(S,axis=1).reshape(n-k,1)
                l=MIN(np.abs(L))
                L=np.sign(L/S)*l
                lin=(lin+np.nansum(L,axis=0)).reshape(1,n)
                L=lin-L
            d_bits[model]=demod(lin)
    return d_bits


# In[206]:


def encode(msg):
    G=np.array([[1,0,0,0,1,0],[0,1,0,0,0,1],[0,0,1,0,1,0],[0,0,0,1,0,1]])
    msg_en=np.dot(msg,G)%2
    r_x=np.array([1 if x==0 else -1 for x in msg_en])
    r_x=r_x+np.random.normal(0,std,(r_x.shape[0],))
    return r_x,msg_en


# In[207]:


ITER=100
MX_STD=10
std=0
bler=[[0 for i in range(MX_MODEL)] for j in range(MX_STD)]


# In[208]:


for iter__ in range(1,MX_STD):
    std=iter__/10
    for iter_ in range(ITER):
        msg=[np.random.randint(0,2) for i in range(k)]
        r_x,msg_en=encode(msg)
        d_bits=decode(H,r_x)
        for model in models:
            if (d_bits[model]==msg_en).all()==True:
                bler[iter__][model]+=1
    print('STD ->',std)
    for model in models:
        bler[iter__][model]/=ITER
        bler[iter__][model]=1-bler[iter__][model]
        print('Model No ->',model,' = ',bler[iter__][model]*100,'%')
    print('----------')

