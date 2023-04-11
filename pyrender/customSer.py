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

def from_text_list(fn:str):
    #TODO
    raise Exception("Unimplemented")
    with open(fn) as f:
        dims = list(map(int,f.readline().strip().split(' ')[1:]))
        l="aa"
        data=[]
        while(len(l)>1):
            l=f.readline()
            (data.append(x) for x in list(map(float,l.strip().split(' '))))
        r=torch.tensor(l)
        return r.cuda().reshape(dims)

def to_text_list(t:list[torch.Tensor],fn:str):
    #TODO
    raise Exception("Unimplemented")
    with open(fn,"w") as f:
        dims=t.size()
        f.write(str(len(dims))+' ')
        for dim in dims:
            f.write(str(dim)+' ')
        f.write('\n')
        for item in t.cpu().reshape(t.numel()):
            f.write(str(item)+' ')

def from_bin_list(fn:str):
    with open(fn,"rb") as f:
        nts=struct.calcsize("q")
        nt,=struct.unpack("q",f.read(nts))
        t:list[torch.Tensor]=[]
        for i in range(nt):
            qs=struct.calcsize("q")
            n_dim:int=struct.unpack("q",f.read(qs))[0]
            qs=struct.calcsize("q"*n_dim)
            dims=struct.unpack("q"*n_dim,f.read(qs))
            expected = reduce((lambda x, y: x * y), dims)
            data = bytearray(f.read(expected*struct.calcsize('f')))#bytes is not mutable, which pytorch does not like.
            ret = torch.frombuffer(data,dtype=torch.float32)
            assert(ret.numel()%expected==0)
            t.append(ret.cuda().reshape(dims))
        if(f.read(1)!=b''):
            raise Exception('File was not empty after all specified tensors were read!')
        return t

def to_bin_list(tl:list[torch.Tensor],fn:str):
    with open(fn,"wb") as f:
        f.write(struct.pack("q",len(tl)))
        for t in tl:
            dims=t.size()
            f.write(struct.pack("q",len(dims)))
            f.write(struct.pack("q"*len(dims),*dims))
            f.write(t.detach().cpu().to(torch.float32).numpy().tobytes())
