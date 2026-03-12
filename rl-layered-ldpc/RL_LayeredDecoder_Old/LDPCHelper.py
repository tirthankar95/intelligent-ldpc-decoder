import numpy as np
import Global as Global

#This function is used in MinSum decoder.
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
    A=["" for i in range(Global.Layers) ];Str=""
    for i in range(Global.Layers):
        A[i]=str(i)
    cnt=0
    StrToNo={}
    NoToStr=[ 0 for i in range(Global.NStates) ]
    fn(A,"",0,Global.subGroups)
    
if __name__=="__main__":
    init()
    assert(cnt==Global.NStates)