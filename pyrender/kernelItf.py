from cuda import nvrtc
import cuda
import os
import cuda_kernels
import torch
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda
import math
import args

max_threads=int(256)
line_max_threads=(max_threads,int(1),int(1))
square_max_thread_size: tuple[int, int, int] = (*map(int,2*[math.sqrt(max_threads)]+[1]),)
cuda_context=None
def initialize():
    global max_threads,line_max_threads,square_max_thread_size,cuda_context
    
    #NOTE: the import below overrides the torch's CUDA environment and causes errors.
    #import pycuda.autoinit as cuda_context 
    #We do this instead, but note that torch has to be imported beforehand either from this file or another.
    drv.init()
    cuda_context = drv.Device(0).retain_primary_context()
    cuda_context.push()
    

    device = drv.Device(0)
    device_data= device.get_attributes()
    
    max_threads=int(device_data[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK])
    line_max_threads=(max_threads,1,1)
    square_max_thread_size = (*map(int,2*[math.sqrt(max_threads)]+[1]),)

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

constructed_modules={}
def prepend_includes(txt:str,loaded:set=None):
    if(loaded==None):
        loaded=set()
    prep=""
    while(txt.startswith("#include \"")):
        idx=txt.find('\n')
        inc=txt[:idx]
        v=inc.split('"')[1]
        if(os.path.exists(f"cuda_kernels/{v}")):
            if(not v in loaded):
                loaded.add(v)
                prep=prep+prepend_includes(open(f"cuda_kernels/{v}").read(),loaded)
            txt=txt[idx+1:]
        else:
            break
    return prep+txt

def assemble_source(m:str):
    d=m.split(' ')
    src = open(f"cuda_kernels/{d[0]}.cu").read()
    MACROS=d[1:]
    macros="".join(
        ["#define "+i.replace('=',' ')+'\n' for i in MACROS]
    )
    #Here we would put headers if we had any important ones
    #if(d[0]=='plot'):
        #src = macros+open(f"cuda_kernels/cameras.cuh").read()+'\n'+open(f"cuda_kernels/environments.cuh").read()+'\n'+open(f"cuda_kernels/{d[0]}.cu").read()
    #else:
    src = macros+prepend_includes(open(f"cuda_kernels/{d[0]}.cu").read())
    return src
from pycuda._driver import Function as pyf
import threading
COMPILE_LOCK=threading.Lock()
VIEW_LOCK=threading.Lock()
#Due to the mutexes, this function is an absolute mess.
def get_kernel(m:str,f:str) -> pyf:
    try:
        VIEW_LOCK.acquire()
        has_view_lock=True
        global constructed_modules
        if m in constructed_modules:
            return constructed_modules[m][0].get_function(f)
        else:
            try:
                VIEW_LOCK.release()
                has_view_lock=False
                COMPILE_LOCK.acquire()#Makes sure kernels are not compiled any more than once
                if m in constructed_modules:
                    return constructed_modules[m][0].get_function(f)
                #TODO? caching?
                print(f"COMPILING CUDA MODULE \"{m}\"")
                #JIT
                src=assemble_source(m)
                try:
                    mod = SourceModule(
                        src
                    )
                except Exception as e:
                    #.vscode/ is an ok place to hide them.
                    print("Cuda compilation errors occurred. dumped kernel.cu and errors.txt in ./.vscode/")
                    with open(".vscode/kernel.cu","w") as ff:
                        ff.write(src)
                    with open(".vscode/errors.txt","w") as ff:
                        ff.write(str(e))
                    raise e
                VIEW_LOCK.acquire()
                has_view_lock=True
                constructed_modules[m]=(mod,threading.current_thread().name)
                fun = mod.get_function(f)
                print(f"CUDA MODULE \"{m}\" COMPILED!")
                return fun
            finally:
                COMPILE_LOCK.release()
    finally:
        if(has_view_lock):VIEW_LOCK.release()

from pycuda.gpuarray import GPUArray
import numpy as np
def null_array():
    return GPUArray((0,), dtype=np.float32,
                         gpudata=0)
def gpu_array(tensor):
    return GPUArray(tensor.shape, dtype=np.float32,
                         gpudata=tensor.data_ptr())

LAST_DBG=0

def plotSinglePointsToTensor(cam_type:int,cam_data:(torch.Tensor or list[float]), points:torch.Tensor, environment_type:int, environment:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #TODO: remove some of the debug stuff.
    # global LAST_DBG
    # DBG_POSITIONS=torch.zeros((2*points.size(0),),dtype=torch.int32,device='cuda').contiguous()
    cam_type=int(cam_type)
    ndim=points.shape[-1]-3
    w,h=map(int,cam_data[2:4])
    if(type(cam_data)==list):
        gpu_cam_data=torch.tensor(cam_data).to(torch.float32).cuda().contiguous()
    else:
        gpu_cam_data=cam_data.clone().detach().to(torch.float32).cuda().contiguous()
    num_points = points.shape[0]
    
    assert(points.is_contiguous())
    env=environment.cuda().contiguous()


    module_name=f"plot CAM_TYPE={cam_type} NDIM={ndim} STRUCTURAL_REFINEMENT={int(args.STRUCTURAL_REFINEMENT)} ENVIRONMENT_TYPE={environment_type}"
    plot_points = get_kernel(module_name,"plot")
    determine_depth = get_kernel(module_name,"determine_depth")
    bundle = get_kernel(module_name,"bundle")
    #plot_background = get_kernel(module_name,"background")
    plot = torch.zeros([h,w,ndim+1],device='cuda').contiguous()
    weights = torch.zeros([h,w],device='cuda').contiguous()
    torch.cuda.synchronize()#This way it is reported if the ones above are busted
    plot[::,::,-1]=1e9#initialize to far plane.
    
    determine_depth(
        gpu_array(plot),
        gpu_array(points),
        np.int32(num_points),
        gpu_array(gpu_cam_data),
        grid=(1+((num_points-1)//max_threads),1,1),
        block=line_max_threads
    )
    drv.Context.synchronize()
    plot_points(
        gpu_array(plot),
        gpu_array(weights),
        gpu_array(points),
        np.int32(num_points),
        gpu_array(gpu_cam_data),
        # gpu_array(DBG_POSITIONS),
        grid=(1+((num_points-1)//max_threads),1,1),
        block=line_max_threads
    )
    drv.Context.synchronize()
    bundle(
        gpu_array(plot),
        gpu_array(weights),
        gpu_array(env),
        np.int32(h),
        np.int32(w),
        gpu_array(gpu_cam_data),
        grid=(1+(h-1//square_max_thread_size[0]),1+(w-1//square_max_thread_size[1]),1),
        block=square_max_thread_size
    )
    # if(cam_data[2]>=1370):
    #     #print(str(threading.currentThread().getName()),DBG_POSITIONS.float().mean(),DBG_POSITIONS.data_ptr())
    #     LAST_DBG=DBG_POSITIONS
    drv.Context.synchronize()
    return plot,weights

def plotSinglePointsBackwardsToTensor(weights:torch.Tensor,cam_type:int,cam_data:list[float],points:torch.Tensor,environment_type:int,environment:torch.Tensor,plot:torch.Tensor,plot_grad:torch.Tensor):
    #TODO: delete debug stuff.
    # DBG_POSITIONS=torch.zeros((2*points.size(0),),dtype=torch.int32,device='cuda').contiguous()
    #if(cam_data[2]<1370):
    #    return None,None,None
    cam_type=int(cam_type)
    ndim=points.shape[-1]-3
    w,h=map(int,cam_data[2:4])
    if(type(cam_data)==list):
        gpu_cam_data=torch.tensor(cam_data).to(torch.float32).cuda().contiguous()
    else:
        gpu_cam_data=cam_data.clone().detach().to(torch.float32).cuda().contiguous()
    num_points = points.shape[0]
    assert(points.is_contiguous())
    env=environment.cuda().contiguous()
    assert(plot_grad.is_contiguous())
    module_name=f"plot CAM_TYPE={cam_type} NDIM={ndim} STRUCTURAL_REFINEMENT={int(args.STRUCTURAL_REFINEMENT)} ENVIRONMENT_TYPE={environment_type}"
    
    cam_data_grad=torch.zeros_like(cam_data).contiguous()
    points_grad=torch.zeros_like(points).contiguous()
    environment_grad=torch.zeros_like(env).contiguous()
    torch.cuda.synchronize()#This way it is reported if the ones above are busted
    
    #per-plotted point backward
    backward = get_kernel(module_name,"backward")
    #per-pixel backward
    backward_pixel = get_kernel(module_name,"backward_pixel")
    
    #could run in parallel?
    backward(
        gpu_array(cam_data_grad),
        gpu_array(gpu_cam_data),
        gpu_array(points_grad),gpu_array(points), np.int32(num_points),
        gpu_array(plot), gpu_array (plot_grad), gpu_array(weights),
        # gpu_array(DBG_POSITIONS),
        grid=(1+((num_points-1)//max_threads),1,1),
        block=line_max_threads
    )
    # delta=(LAST_DBG-DBG_POSITIONS).float().mean()
    # if(delta!=0.0):
    #     print('b'+str(threading.currentThread().getName()),DBG_POSITIONS.float().mean(),DBG_POSITIONS.data_ptr())
    #     print('bd'+str(threading.currentThread().getName()),(LAST_DBG-DBG_POSITIONS).float().mean(),LAST_DBG.data_ptr())
    
    
    backward_pixel(
        gpu_array(cam_data_grad),
        gpu_array(gpu_cam_data),
        gpu_array(plot), gpu_array (plot_grad), gpu_array(weights),
        np.int32(h),np.int32(w),
        gpu_array(environment_grad),gpu_array(env),
        grid=(1+((num_points-1)//max_threads),1,1),
        block=line_max_threads
    )
    
    drv.Context.synchronize()
    
    if(points_grad.isnan().any()):
        print("BACKWARD RETURNED NANs!")
        return None,None,None
    return cam_data_grad,points_grad,environment_grad

#This almost feels like a hack
def cleanup():
    VIEW_LOCK.acquire()
    cuda_context.pop()
    for m in [*constructed_modules.keys()]:
        if(constructed_modules[m][1]==threading.current_thread().name):
            del constructed_modules[m]
    VIEW_LOCK.release()