****
Backup0/1 ~ Backups.
Current code.
****

Global.py -> This module has global variables used in this project.
Graph.py -> This module creates state transition graph.
LDPCmain.py -> This module has actual LDPC implementation and Reinforcement Learning implementations. 
LDPCHelper.py -> ...

If L -> is the number of layers.
L!+L!/1!+L!/2!..+L!/(L-1)!

If we have 9 layers 0,1,2...8
If we take 3 layers at a time 9C3.
Size of each layer grp = 9/3 =3

**********TODO*********
1. Store the state value in a file.
   When running the code load this state value.
   After running certain examples update the state value.

2. Can you group the codes based on which code responds better to
   which layering. This is similar to the amazon example of the machine
   learning book where similar things have higher dot product.
***********************