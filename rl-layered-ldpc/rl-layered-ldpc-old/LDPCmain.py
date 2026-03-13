import LDPCHelper as LD
import numpy as np
import Graph as Graph
import Global as Global
import warnings

Mat=[]
H=[]
def init():
    global Mat,H
    H=np.random.randint(0,2,(Global.n-Global.k,Global.n))
    Mat=np.split(H,Global.Layers)

def LayeredLDPC0(l_intrinsic,mxIter):
    global Mat
    lin=l_intrinsic
    LMat=[]
    for i in range(mxIter):
        for j in range(Global.Layers):
            if len(LMat)!=0:
                lin=lin-np.nansum(LMat[j],axis=0).reshape(1,Global.n)
            L=np.multiply(Mat[j],lin)
            indx=np.where(L==0)
            L[indx]=np.nan
            S=np.sign(L)
            S=np.nanprod(S,axis=1).reshape(Global.sz,1)
            L=np.sign(L/S)*LD.MIN(np.abs(L))
            lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
            LMat.append(L)
    return lin

# This function is suppose to return the reward.
# l_instrinsic will come from parent
def rewardLDPC1(l_intrinsic,obj):
    lin=l_intrinsic
    action=obj.state[-Global.subGroups:]
    iter_=0
    LMat=[]
    for ch in action:
        j=int(ch)
        if len(obj.LMat)!=0:
            lin=lin-np.nansum(obj.LMat[iter_],axis=0).reshape(1,Global.n)
            iter_=iter_+1
        L=np.multiply(Mat[j],lin)
        indx=np.where(L==0)
        L[indx]=np.nan
        S=np.sign(L)
        S=np.nanprod(S,axis=1).reshape(Global.sz,1)
        L=np.sign(L/S)*LD.MIN(np.abs(L))
        lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
        LMat.append(L)
    obj.LMat=LMat
    obj.codeword=lin
    if np.all(np.dot(H,lin.T)==0):
        return 5
    else:
        return -1

def LayeredLDPC1(obj):
    try:
        p=len(Graph.g.adj[obj.id])
    except Exception:
        return#the end has been encountered.
    for child in Graph.g.adj[obj.id]:
        obj.value+=(1/p)*(rewardLDPC1(obj.codeword,child)+child.value)
    for child in Graph.g.adj[obj.id]:
        LayeredLDPC1(child)

def display(Ostate):
    sample=open("op.txt","w")
    qu, qu1 = [], []
    qu.append(Ostate, id)
    print(Ostate.value,file=sample)
    print("-----------------",file=sample)
    while len(qu)!=0:
        while len(qu)!=0:
            par=qu.pop()
            try:
                Graph.g.adj[par]
            except Exception:
                continue
            for child in Graph.g.adj[par]:
                qu1.append(child.id)
                print(child.value,end=" , ",file=sample)
        if len(qu1)!=0:
            print("\n-----------------",file=sample)
        qu=qu1
        qu1=[]

def Test0():
    global Mat,H
    Global.n=7
    Global.k=3
    Global.Layers=2
    mxIter=2
    H=np.array([[1,1,1,0,1,0,0],[0,0,0,1,0,1,1],[1,1,0,1,0,0,1],[0,0,1,0,1,1,0]])
    Mat=np.split(H,Global.Layers)
    l_intrinsic=np.array([0.2,-0.3,1.2,-0.5,0.8,0.6,-1.1])
    l_intrinsic_R=LayeredLDPC0(l_intrinsic,mxIter)
    print(l_intrinsic_R)
    print("Currently Unavailable.")

if __name__=='__main__':
    if Global.NoWarnings:
        warnings.filterwarnings("ignore")
    init()
    LD.init()
    Graph.counter=0
    l_intrinsic=np.random.rand(1,Global.n)-0.5
    Ostate=Graph.node(0,"",l_intrinsic)
    Graph.build(Ostate)
# Code ...
    LayeredLDPC1(Ostate)
    #This display is to check if the weight has been updated.
    display(Ostate)
# Code ...
    if Global.Test0:
        Test0()
