import EncoderPlusChannel as epc
import HG1 as hg1
import matplotlib.pyplot as plt
import numpy as np
import math
import Global
import threading
from DecoderHelper import ATANH,tanh_mine,atanh_mine,atanh_mine_poly,atanh_mine_NN,genAll,demod,MIN,atanh_NNtrain

alpha=0.5;beta=0.5
ITER=10;lowerLimit=4;mx_iter=5
selModel = [0,1,2,3,4,5]
MODEL=len(selModel)
# 0 -> original model.
# 1 -> alpha beta model.
# 2 -> min sum model
# 3 -> tnh_atnh_approx model.
# 4 -> atnh polynomial model.
# 5 -> atnh NN model.
# 6 -> default model
##############################################################
def decode(H,c_Rx,std):
    l_intrinsic = np.multiply(2 / std**2, c_Rx)
    d_bits=[[] for i in range(MODEL)]
    for model in selModel:
        lin = l_intrinsic
        L = np.multiply(H, lin)
        indx = np.where(L == 0)
        L[indx] = np.nan
        if model==0: # 0 -> original model.
            for i in range(mx_iter):
                L=np.tanh(L/2)
                L_=np.nanprod(L,axis=1).reshape(Global.n-Global.k,1)
                L=np.divide(L_, L)
                L=2*ATANH(L)
                lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
                L=lin-L
            d_bits[model]=demod(lin)
        if model==1: # 1 -> alpha beta model.
            for i in range(mx_iter):
                S=np.sign(L)
                S=np.nanprod(S,axis=1).reshape(Global.n-Global.k,1)
                l=MIN(np.abs(L))
                L=np.sign(L/S)*l
                L=alpha*L+beta
                lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
                L=lin-L
            d_bits[model]=demod(lin)
        if model==2: # 2 -> min sum model
            for i in range(mx_iter):
                S=np.sign(L)
                S=np.nanprod(S,axis=1).reshape(Global.n-Global.k,1)
                l=MIN(np.abs(L))
                L=np.sign(L/S)*l
                lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
                L=lin-L
            d_bits[model]=demod(lin)            
        if model==3: # 3 -> tnh_atnh_approx model.
            for i in range(mx_iter):
                L = tanh_mine(L/2)
                L_ = np.nanprod(L, axis=1).reshape(Global.n - Global.k, 1)
                L = np.divide(L_, L)
                L = 2*atanh_mine(L)
                lin = (lin + np.nansum(L, axis=0)).reshape(1, Global.n)
                L = lin - L
            d_bits[model] = demod(lin)
        if model==4: # 4 -> atnh polynomial model.
            for i in range(mx_iter):
                L=np.tanh(L/2)
                L_ = np.nanprod(L, axis=1).reshape(Global.n - Global.k, 1)
                L = np.divide(L_, L)
                L = 2*atanh_mine_poly(L)
                lin = (lin + np.nansum(L, axis=0)).reshape(1, Global.n)
                L = lin - L
            d_bits[model] = demod(lin)
        if model==5: # 5 -> atnh NN model.
            for i in range(mx_iter):
                L=np.tanh(L/2)
                L_ = np.nanprod(L, axis=1).reshape(Global.n - Global.k, 1)
                L = np.divide(L_, L)
                L = 2*atanh_mine_NN(L)
                lin = (lin + np.nansum(L, axis=0)).reshape(1, Global.n)
                L = lin - L
            d_bits[model] = demod(lin)
        if model==6: # 6 -> default model
            d_bits[model] = demod(l_intrinsic)
    return np.array(d_bits)
#################### PARALLEL THREADS ########################
class myThread(threading.Thread):
    def __init__(self,threadID,name):
        threading.Thread.__init__(self)
        self.threadID=threadID # std value.
        self.name=name
    def run(self):
        #threadLock.acquire()
        RUN(self.threadID,self.name)
        #threadLock.release()

def RUN(i_upper,name):
    print('---',i_upper,' ~ ',name,'---')
    std=i_upper/10
    alpha, beta = mem[i_upper-1][0],mem[i_upper-1][1]#ml.train()
    c_Rx = epc.rx_message(c_encoded,std)
    for i in range(c_Rx.shape[0]):
        dec_bits=decode(H,c_Rx[i],std) ## get results for all the models.
        for model in selModel: # FOR DIFF models check the accuracy.
            if (dec_bits[model]==c_encoded[i]).all()==True:
                Global.code_err[model][i_upper - lowerLimit] +=1
# Calculating the BLER
    for model in range(MODEL):
        Global.code_err[model][i_upper-lowerLimit]=1-Global.code_err[model][i_upper-lowerLimit]/c_Rx.shape[0]
######################## MAIN ################################
tmp=[]
atanh_NNtrain()
genAll(1000)
H,c_encoded=hg1.encode(Global.msg)
#Decoding
Global.code_err=[[0 for i in range(lowerLimit,ITER)] for j in range(MODEL)]
#Global.bit_err=[[0 for i in range(1,ITER)] for j in range(MODEL)]
mem=[(1.5014,-0.0022),(1.174,-0.055),(0.8513,0.0464),(0.6218,-0.0205),(0.4713,-0.008),(0.3703,-0.0255),(0.2715,0),(0.3035,-0.0126),(0.2597,0.0063)]
threads=[]

for i_upper in range(lowerLimit,ITER):# FOR DIFF std
    tmpName="Thread-"+str(i_upper)
    threads.append(myThread(i_upper,tmpName))

for t in threads:
    t.start()

for t in threads:
    t.join()
####################### Plotting the results #################################
x_axis=np.array([10*math.log10((100/(i**2))) for i in range(lowerLimit,ITER) ])
color=["rx-","bo-","g^-","kx-","yo-","c^-","mx-"]
for model in selModel:
    plt.plot(x_axis,np.array(Global.code_err[model]),color[model])
plt.legend(('Original','ML-method','Min_sum','tnh_atnh approx','atnh_poly','atnh_NN'))
plt.xlabel('SNR(db)')
plt.ylabel('BLER (n=%d/k=%d)'%(Global.n,Global.k))
#plt.savefig("output.png")
plt.show()
######################################################################
