from cuda import nvrtc
import cuda
import cuda_kernels
import torch
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda
import math

#NOTE: the import below overrides the torch's CUDA environment and causes errors.
#import pycuda.autoinit as cuda_context 
#We do this instead, but note that torch has to be imported beforehand either from this file or another.
drv.init()
cuda_context = drv.Device(0).retain_primary_context()
cuda_context.push()

max_threads=int(256)
line_max_threads=(max_threads,int(1),int(1))
square_max_thread_size: tuple[int, int, int] = (*map(int,2*[math.sqrt(max_threads)]+[1]),)
def init_thread_data():
    device = drv.Device(0)
    device_data= device.get_attributes()
    global max_threads,line_max_threads,square_max_thread_size
    max_threads=int(device_data[pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK])
    line_max_threads=(max_threads,1,1)
    square_max_thread_size = (*map(int,2*[math.sqrt(max_threads)]+[1]),)
init_thread_data()

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
def assemble_source(m:str):
    d=m.split(' ')
    src = open(f"cuda_kernels/{d[0]}.cu").read()
    MACROS=d[1:]
    macros="".join(
        ["#define "+i.replace('=',' ')+'\n' for i in MACROS]
    )
    #Here we would put headers if we had any important ones    
    src = macros+open(f"cuda_kernels/cameras.cuh").read()+'\n'+open(f"cuda_kernels/{d[0]}.cu").read()
    return src
from pycuda._driver import Function as pyf
def get_kernel(m:str,f:str) -> pyf:
    global constructed_modules
    if m in constructed_modules:
        return constructed_modules[m].get_function(f)
    else:
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
        constructed_modules[m]=mod
        fun = mod.get_function(f)
        return fun

from pycuda.gpuarray import GPUArray
import numpy as np

def gpu_array(tensor):
    return GPUArray(tensor.shape, dtype=np.float32,
                         gpudata=tensor.data_ptr())

def plotSinglePointsToTensor(cam_type:int,cam_data:(torch.Tensor or list[float]), points:torch.Tensor, env:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cam_type=int(cam_type)
    ndim=points.shape[-1]-3
    w,h=map(int,cam_data[2:4])
    if(type(cam_data)==list):
        gpu_cam_data=torch.tensor(cam_data).cuda()
    else:
        gpu_cam_data=cam_data.clone().detach().cuda()
    num_points = points.shape[0]
    
    module_name=f"plot CAM_TYPE={cam_type} NDIM={ndim}"
    plot_points = get_kernel(module_name,"plot")
    determine_depth = get_kernel(module_name,"determine_depth")
    bundle = get_kernel(module_name,"bundle")
    #plot_background = get_kernel(module_name,"background")
    plot = torch.zeros([h,w,ndim+1],device='cuda').contiguous()
    weights = torch.zeros([h,w],device='cuda').contiguous()
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
        grid=(1+((num_points-1)//max_threads),1,1),
        block=line_max_threads
    )
    drv.Context.synchronize()
    bundle(
        gpu_array(plot),
        gpu_array(weights),
        np.int32(h),
        np.int32(w),
        grid=(1+(h-1//square_max_thread_size[0]),1+(w-1//square_max_thread_size[1]),1),
        block=square_max_thread_size
    )
    drv.Context.synchronize()
    return plot,weights

def plotSinglePointsBackwardsToTensor(*args):
    #TODO
    return None,None,None

