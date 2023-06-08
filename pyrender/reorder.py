import args
import torch
from time import time
from struct import pack, unpack
from itertools import chain
import random

def reorder(data:torch.Tensor):
    if(args.reorder_points == 'no'):
        return data
    elif(args.reorder_points == 'shuffled'):
        key = lambda x:random.random()
    elif(args.reorder_points == 'morton' or args.reorder_points == 'shuffled_morton'):
        key = morton_code_int
    elif(args.reorder_points == 'morton_float' or args.reorder_points == 'shuffled_morton_float'):#This is not mentioned in the paper as it is basically a bug.
        key = morton_code_float
    else:
        raise Exception(f"point reordering type {args.refine_points} unknown")
    
    start_time=time()
    print(f"STARTING sort of {len(data)} points.")

    original_device=data.device
    cpu_data=data.cpu()
    
    cpu_data = sorted(cpu_data, key = key)
    
    if(args.reorder_points == 'shuffled_morton' or args.reorder_points == 'shuffled_morton_float'):
        block_size=256#for best results, this is set to kernelItf.max_threads probably (to be tested)
        blocks=[cpu_data[i:i+block_size] for i in range(0, len(cpu_data), block_size)]
        random.shuffle(blocks)
        cpu_data=[*chain(*blocks)]
    #FUsed to generate figure 3.4
    # for i in range(len(cpu_data)):
        #cpu_data[i][3:6]=i/len(cpu_data)
        #if(i%256==0):r,g,b=random.random(),random.random(),random.random()
        #cpu_data[i][3+0]=r
        #cpu_data[i][3+1]=g
        #cpu_data[i][3+2]=b
    print("ENDING sort.")
    global idx;idx=0
    rez=torch.stack(cpu_data).to(original_device)
    end_time=time()
    print(f"Time taken:{end_time-start_time:.2f}s")
    return rez
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

def part1by2_32_3(n:int):
    #n&=par128#possibly not required?
    #some things are def. not required here
    n=(n|(n << 32))&par32
    n=(n|(n << 16))&par16
    n=(n|(n <<  8))&par8
    n=(n|(n <<  4))&par4
    n=(n|(n <<  2))&par2
    return n

def pun_l(t):
    return int(unpack('Q',pack('d', t))[0])
def pun_d(t):
    return float(unpack('d',pack('Q', t))[0])

def shl(t,sh):
    if(sh<0):
        return t << -sh
    else:
        return t >> sh

def fetchExp(e):
    return ((pun_l(e)&0b0111111111110000000000000000000000000000000000000000000000000000)>>52)-0b01111111111


def mortonCorrect(x,y,z):
    c0=1*(x<0)+2*(y<0)+4*(z<0)
    exp=max(map(fetchExp,(x,y,z)))
    p2=pow(2,exp)
    pe=(1<<16)/p2
    ix,iy,iz=[int(abs(i)*pe) for i in [x,y,z]]
    data=float(part1by2_32_3(ix)<<2|(part1by2_32_3(iy)<<1)|(part1by2_32_3(iz)))*p2
    return (c0,data)
def mortonFast(x,y,z):#It is only *mildly* faster
    x,y,z=[int(unpack('l',pack('f', t))[0]) for t in  [-x,-y,-z]]
    code=part1by2_32_3(x)<<2|(part1by2_32_3(y)<<1)|(part1by2_32_3(z))
    return abs(float(unpack('d',pack('Q', 0xFFFFFFFFFFFFFFFF & code))[0]))

idx=0
def morton_code_int(data:torch.Tensor):
    x,y,z,*ignored=map(float,data)
    global idx
    idx+=1
    if(idx%100000==0):print(idx)
    return mortonCorrect(x,y,z)
def morton_code_float(data:torch.Tensor):
    x,y,z,*ignored=map(float,data)
    global idx
    idx+=1
    if(idx%100000==0):print(idx)
    return mortonCorrect(x,y,z)
    

