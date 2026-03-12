import LDPCHelper as LD
import Global as Global

class node:
    def __init__(self,ID,STATE,CODE):
        self.id=ID
        self.state=STATE
        self.codeword=CODE
        self.value=0
        self.LMat=[]
        
class graph:
    def __init__(self):
        self.adj={}
    def addEdge(self,u,v):
        if u.id not in self.adj:
            self.adj[u.id]=[]
        self.adj[u.id].append(v)                

counter=0
g=graph()
def build(obj):
    global g,counter
    if(len(obj.state)==Global.Layers):
        return
    tmp={}
    for i in obj.state:
        tmp[i]=True
    for i in LD.NoToStr:
        isPresent=False
        for j in i:
            if j in tmp:
                isPresent=True
                break
        if isPresent==False:
            counter=counter+1
            O=node(counter,obj.state+i,[0 in range(Global.n)])
            g.addEdge(obj,O)
            build(O)

if __name__=="__main__":
    LD.init()
    O=node(counter,"",[0 in range(Global.n)])
    build(O)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    