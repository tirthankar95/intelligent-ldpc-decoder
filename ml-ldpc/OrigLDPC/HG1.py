import numpy as np
import random as r
import Global

def encode(msg):
    K=Global.k
    N=Global.n
    H=np.array([])
    with open('Parity.txt') as f:
        lines=f.readlines()
        for line in lines:
            myarr=np.fromstring(line,dtype=float,sep=" ")
            if len(H)==0:
                H=myarr
            else:
                H=np.vstack([H,myarr])
    n_=len(msg)
    for iter in range(n_):
        parity = np.array([0 for i in range(N - K)])
        with open('Generator.txt') as f:
            lines=f.readlines()
            i=0
            for line in lines:
                myarr=np.fromstring(line,dtype=float,sep=" ")
                c=len(myarr)
                for j in range(c):
                    parity[i]=( parity[i] + msg[iter][ int(myarr[j]) ] )%2
                msg[iter]=np.append(msg[iter],parity[i])
                i=i+1
    msg=np.array(msg)
    Zzz=np.dot(H,msg[iter].T)%2
    flag=0
    for x in Zzz:
        if x!=0:
            print('Incorrect')
            flag=1
            break
    if flag==0:
        print('Correct')
    return H,msg