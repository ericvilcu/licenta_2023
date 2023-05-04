import args
import torch

from struct import pack, unpack
from itertools import chain
import random

def reorder(data:torch.Tensor):
    if(args.reorder_points == 'no'):
        return data
    elif(args.reorder_points == 'shuffled'):
        key = lambda x:random.random()
    elif(args.reorder_points == 'morton' or args.reorder_points == 'shuffled_morton'):
        key = morton_code
    else:
        raise Exception(f"point reordering type {args.refine_points} unknown")
    
    print(f"STARTING sort of {len(data)} points.")

    original_device=data.device
    cpu_data=data.cpu()
    
    cpu_data = sorted(cpu_data, key = key)
    
    if(args.reorder_points == 'shuffled_morton'):
        block_size=256#for best results, this is set to kernelItf.max_threads probably (to be tested)
        blocks=[cpu_data[i:i+block_size] for i in range(0, len(cpu_data), block_size)]
        random.shuffle(blocks)
        cpu_data=[*chain(*blocks)]
    
    print("ENDING sort.")
    global idx;idx=0
    return torch.stack(cpu_data).to(original_device)
def morton_code_SLOW(data:torch.Tensor):
    #VERY SLOW.
    #also think about normalization
    x,y,z,*ignored=map(float,data)
    x,y,z=[bin(int(unpack('Q',pack('d', t))[0]))[2:] for t in  [-x,-y,-z]]
    ml=max(len(t) for t in [x,y,z])
    x,y,z= ['0'*(ml-len(t))+t for t in [x,y,z]]
    return int(''.join(chain(*zip(x,y,z))),2)



#part2  = int('001'*64,2)
#thanks to https://github.com/trevorprater/pymorton/blob/master/pymorton/pymorton.py for providing inspiration for this
from math import ceil
TOT_BITS=64*3
MAX_N=len(bin(64*3)[2:])-1
part_n={(2**(x+1)):int(('0'*(2*(2**x))+'1'*(2**x))*int(ceil(TOT_BITS/(3*(2**x)))),2)
    for x in range(MAX_N)
}
par2  =part_n[2]
par4  =part_n[4]
par8  =part_n[8]
par16 =part_n[16]
par32 =part_n[32]
par64 =part_n[64]
par128=part_n[128]
def part1by2_64_3(n:int):
    #n&=par128#possibly not required?
    n=(n|(n << 64))&par64
    n=(n|(n << 32))&par32
    n=(n|(n << 16))&par16
    n=(n|(n <<  8))&par8
    n=(n|(n <<  4))&par4
    n=(n|(n <<  2))&par2
    return n

idx=0
def morton_code(data:torch.Tensor):
    #Still kinda slow but not as slow. I need to check if I can somehow move this calculation to the GPU. probably not if I want to use 256 big ints; maybe if I reduce them to 64 bits somehow
    x,y,z,*ignored=map(float,data)
    x,y,z=[int(unpack('Q',pack('d', t))[0]) for t in  [-x,-y,-z]]
    
    global idx
    idx+=1
    if(idx%100000==0):print(idx)
    
    code=part1by2_64_3(x)<<2|(part1by2_64_3(y)<<1)|(part1by2_64_3(z))
    #code_slow=morton_code_SLOW(data)
    #assert(code==code_slow)
    return code
    

