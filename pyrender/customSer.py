#simple serialization
import torch
import numpy as np
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
        
        