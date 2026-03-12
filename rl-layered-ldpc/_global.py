mxSize=100
FACTORIAL=[0 for i in range(mxSize)]
def factorial(n):
    global FACTORIAL
    if n==0 or n==1:
        return 1
    if FACTORIAL[n]!=0:
        return FACTORIAL[n]
    FACTORIAL[n]=n*factorial(n-1)
    return FACTORIAL[n]

Layers=4
subGroups=1
NStates=int(factorial(Layers)/factorial(Layers-subGroups))
NStates=int(NStates/factorial(subGroups))
#initial number of groups of 3.
n=8
k=4
Test0=False
Test1=False
NoWarnings=True
ITERATIONS=True
sz=(n-k)//Layers # number of rows in each layer.
mxIter=1
Episodes=1
optAll=1
LIMIT=2**8
sigma=0.5
inf=100000
'''
Layers=9
subGroups=3
n=15
k=6
mxIter=15
Episodes=1
optAll=1
LIMIT=64
sigma=0.6
'''
