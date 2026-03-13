import util as LD
import numpy as np
import graph as graph
import _global as G
import time
import warnings

Mat=[]
H=[]
def init():
    global Mat,H
    H=np.array([[1,0,0,1,0,0,1,0],[0,1,0,0,1,0,0,1],[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]])
    Mat=np.split(H, G.Layers)

def demod(codeword1):
    codeword=codeword1.copy()
    indx0=np.where(codeword>0)
    indx1=np.where(codeword<=0)
    codeword[indx0]=0
    codeword[indx1]=1
    return codeword

def LayeredLDPC0():
    global Mat,l_intrinsic
    lin=l_intrinsic
    LMat=[[0 for j in range(G.n)] for i in range(G.Layers)]
    for i in range(G.mxIter):
        for j in range(G.Layers):
            lin=lin-LMat[j]
            L=np.multiply(Mat[j],lin)
            indx=np.where(L==0)
            L[indx]=np.nan
            S=np.sign(L)
            S=np.nanprod(S,axis=1).reshape(G.sz,1)
            L=np.sign(L/S) * LD.MIN(np.abs(L))
            lin=(lin+np.nansum(L,axis=0)).reshape(1, G.n)
            LMat[j]=np.nansum(L,axis=0).reshape(1, G.n)
    res=np.dot(H,demod(lin).T)%2
    return res,lin

# This function is suppose to return the reward.
# l_instrinsic will come from parent
finIntrinsic=[]
flag=False
def rewardLDPC1(state):
    global l_intrinsic,codewordCorrected,flag
    lin=l_intrinsic
    start, reward = 0, 0
    LMat=[[0 for j in range(G.n)] for i in range(G.Layers)]
    for iter_ in range(G.mxIter):
        while start < G.Layers:
            action=state[start:start+G.subGroups]
            start+=G.subGroups
            for ch in action:
                j=int(ch)
                lin=lin-LMat[j]
                L=np.multiply(Mat[j],lin)
                indx=np.where(L==0)
                L[indx]=np.nan
                S=np.sign(L)
                S=np.nanprod(S,axis=1).reshape(G.sz,1)
                L=np.sign(L/S)*LD.MIN(np.abs(L))
                lin=(lin+np.nansum(L,axis=0)).reshape(1,G.n)
                LMat[j]=np.nansum(L,axis=0).reshape(1,G.n)
        res=np.dot(H,demod(lin).T)%2
        if np.all(res==0):
            reward=reward+5
            if flag:
                codewordCorrected.append(demod(lin[0]))
                flag=False
            break
        else:
            reward=reward-1
    if Test1:
        return res,lin
    else:
        return reward

def LayeredLDPC1(obj):
    try:
        p=len(graph.g.adj[obj.id])
    except Exception:
        obj.value=rewardLDPC1(obj.state)
        return obj.value
    for child in graph.g.adj[obj.id]:
        obj.value+=(1/p)*(child.value+LayeredLDPC1(child))
    return obj.value

def Test0():
    global Mat,H,l_intrinsic
    print('Test0')
    G.Layers=2 #This is the number of G.Layers in H-matrix.
    # H = (n-k)*n
    H=np.array([[1,1,1,0,1,0,0],[0,0,0,1,0,1,1],[1,1,0,1,0,0,1],[0,0,1,0,1,1,0]])
    Mat=np.split(H,G.Layers)
    l_intrinsic=np.array([0.2,-0.3,1.2,-0.5,0.8,0.6,-1.1])
    result,l_intrinsic_R=LayeredLDPC0() # After layered decoding this function
    # returns the final decoded result and the finnal l_intrinsic_R
    print('Before:\n',demod(l_intrinsic))
    print('Final Result:\n',result.T,'\n',l_intrinsic_R)

def Test1():
    global Mat,H,l_intrinsic
    print('Test1')
    G.Layers=2
    Str="01"
    H=np.array([[1,1,1,0,1,0,0],[0,0,0,1,0,1,1],[1,1,0,1,0,0,1],[0,0,1,0,1,1,0]])
    Mat=np.split(H,G.Layers)
    l_intrinsic=np.array([0.2,-0.3,1.2,-0.5,0.8,0.6,-1.1])
    result,l_intrinsic_R=rewardLDPC1(Str)
    print('Before:\n',demod(l_intrinsic))
    print('Final Result:\n',result.T,'\n',l_intrinsic_R)

def clear(obj):
    obj.value=0
    try:
        len(graph.g.adj[obj.id])
        for child in graph.g.adj[obj.id]:
            clear(child)
    except Exception:
        return

def getMax(obj):
    global mx
    try:
        len(graph.g.adj[obj.id])
        for child in graph.g.adj[obj.id]:
            getMax(child)
    except Exception:
        if mx<obj.value:
            mx=obj.value
        return

SStr=""
def extractUtil(obj):
    global mx,SStr
    try:
        len(graph.g.adj[obj.id])
        for child in graph.g.adj[obj.id]:
            extractUtil(child)
    except Exception:
        if obj.value==mx:
            SStr+='1'
        else:
            SStr+='0'

def extract(obj):
    global mx,SStr
    getMax(obj)
    SStr=""
    extractUtil(obj)
    return SStr

if __name__=='__main__':
    if G.NoWarnings:
        warnings.filterwarnings("ignore")
    if Test0:
        Test0()
    elif Test1:
        Test1()
    else:
    #...
        init()
        LD.init()
        graph.counter=0
        np.random.seed(int(time.time()))
        if G.optAll!=1:
            l_intrinsic=np.random.rand(1,G.n)-0.5
            Ostate=graph.node(0,"")
            graph.build(Ostate,Ostate.state)
            for G.Episodes in range(G.Episodes):
                LayeredLDPC1(Ostate)
            LD.display0(Ostate)
        else:
            LD.lim, LD.codeword, number = 0, [], []
            LD.genAll([])
            codewordNp=np.array(LD.codeword)
            noise=np.random.normal(0,G.sigma,(G.LIMIT,G.n))
            codewordRx=LD.modulate(codewordNp)+noise
            #codewordRx is the received codeword
            #You have to send the array codewordRx to the LDPC decoder
            Ostate=graph.node(0,"")
            graph.build(Ostate, Ostate.state)
            codewordCorrected=[]
            cnt=0
            for iter_ in range(G.LIMIT):
                for G.Episodes in range(G.Episodes):
                    l_intrinsic=codewordRx[iter_]
                    flag=True
                    LayeredLDPC1(Ostate)
                    if flag:
                        codewordCorrected.append(demod(codewordRx[iter_]))
                        cnt=cnt+1
                        flag=False
                    mx=-G.inf
                    number.append(extract(Ostate))
                    clear(Ostate)
            LD.display1(H,codewordCorrected,codewordRx,number)
            print(cnt,"These number of codewords cannot be decoded.")
    #...
