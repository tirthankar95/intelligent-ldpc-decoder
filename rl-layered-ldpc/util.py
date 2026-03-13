import numpy as np
import _global as G
import graph

#This function is used in MinSum decoder.
def MIN(L):
    (r,c)=L.shape
    L[np.isnan(L)]=999
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

cnt=0
def swap(A,a,b):
    tmp=A[a]
    A[a]=A[b]
    A[b]=tmp

def fn(Str,Sol,strt,k):
    global cnt,NoToStr,StrToNo
    if k==0:
        NoToStr[cnt]=Sol
        StrToNo[Sol]=cnt
        cnt=cnt+1
        return
    for i in range(strt,len(Str)):
        fn(Str,Sol+Str[i],i+1,k-1)

def init():
    global cnt,NoToStr,StrToNo
    A=["" for i in range(G.Layers) ]
    for i in range(G.Layers):
        A[i]=str(i)
    cnt=0
    StrToNo={}
    NoToStr=[ 0 for i in range(G.NStates) ]
    fn(A,"",0,G.subGroups)

lim, codeword = 0, []
def genAll(transitional):
    global lim,codeword
    if lim==G.LIMIT:
        return
    if len(transitional)==G.n:
        tmp=transitional[:]
        codeword.append(tmp)
        lim=lim+1
        return
    transitional.append(0)
    genAll(transitional)
    transitional.pop()
    transitional.append(1)
    genAll(transitional)
    transitional.pop()

def modulate(codeword):
    Arr=codeword.copy()
    indx0=np.where(Arr==0)
    indx1=np.where(Arr==1)
    Arr[indx0]=1
    Arr[indx1]=-1
    return Arr

def display0(Ostate):
    sample=open("op.txt","w")
    qu=[], qu1 = [], []
    qu.append(Ostate.id)
    print(Ostate.value,file=sample)
    print("-----------------",file=sample)
    while len(qu)!=0:
        while len(qu)!=0:
            par=qu.pop()
            try:
                graph.g.adj[par]
            except Exception:
                continue
            for child in graph.g.adj[par]:
                qu1.append(child.id)
                print(child.value,end=" , ",file=sample)
        if len(qu1)!=0:
            print("\n-----------------",file=sample)
        qu=qu1
        qu1=[]


def display1(H,Ocode,Rcode,number):
    sample=open("op1.txt","w")
    print("H-Matrix",file=sample)
    print(H,file=sample)
    print()
    print("-------------------------------------------------------------",file=sample)
    print("|      |            |            |                            |",file=sample)
    print("|Sr No.| Corr. Code |   Rx. Code |           Number           |",file=sample)
    print("|      |            |            |                            |",file=sample)
    print("--------------------------------------------------------------",file=sample)
    n=Rcode.shape[0]
    OcodeN, RcodeN = [], []
    for i in range(n):
        Str, Str1 = "", ""
        for j in range(n):
            if int(Ocode[i][j])==0:
                Str=Str+'0'
            else:
                Str=Str+'1'
            Str1=Str1+"{:.2f}".format(Rcode[i][j])+" "
        OcodeN.append(Str)
        RcodeN.append(Str1)
    for i in range(n):
        if i>=9:
            print("|  %d. |  %s  |  %s  |  %s  |"%(i+1,OcodeN[i],RcodeN[i],number[i]),file=sample)
        else:
            print("|  %d.  |  %s  |  %s  |  %s  |"%(i+1,OcodeN[i],RcodeN[i],number[i]),file=sample)
    print("--------------------------------------------------------------",file=sample)


if __name__=="__main__":
    init()
    print(cnt)
