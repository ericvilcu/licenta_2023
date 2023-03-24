#simple serialization for tensors
import torch
import struct
from functools import reduce
def from_text(fn:str):
    with open(fn) as f:
        dims = list(map(int,f.readline().strip().split(' ')[1:]))
        l="aa"
        data=[]
        while(len(l)>1):
            l=f.readline()
            (data.append(x) for x in list(map(float,l.strip().split(' '))))
        r=torch.tensor(l)
        return r.cuda().reshape(dims)

def to_text(t:torch.Tensor,fn:str):
    with open(fn,"w") as f:
        dims=t.size()
        f.write(str(len(dims))+' ')
        for dim in dims:
            f.write(str(dim)+' ')
        f.write('\n')
        for item in t.cpu().reshape(t.numel()):
            f.write(str(item)+' ')

def from_bin(fn:str):
    with open(fn,"rb") as f:
        qs=struct.calcsize("q")
        n_dim:int=struct.unpack("q",f.read(qs))[0]
        qs=struct.calcsize("q"*n_dim)
        dims=struct.unpack("q"*n_dim,f.read(qs))
        data = bytearray(f.read())#bytes is not mutable, which pytorch does not like.
        expected = reduce((lambda x, y: x * y), dims)
        ret = torch.frombuffer(data,dtype=torch.float32)
        assert(ret.numel()%expected==0)
        mul=ret.numel()//expected
        if(mul>1):dims=(mul,*dims)
        return ret.cuda().reshape(dims)

def to_bin(t:torch.Tensor,fn:str):
    with open(fn,"wb") as f:
        dims=t.size()
        f.write(struct.pack("q",len(dims)))
        f.write(struct.pack("q"*len(dims),*dims))
        f.write(t.detach().cpu().numpy().tobytes())
        
        
        
        
