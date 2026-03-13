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

Layers=9
subGroups=3
NStates=int(factorial(Layers)/factorial(Layers-subGroups))
NStates=int(NStates/factorial(subGroups))
#initial number of groups of 3.
n=15
k=6
Test0=False
NoWarnings=False
sz=(n-k)//Layers # number of rows in each layer.
