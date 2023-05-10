import args
import torch
import kernelItf as kern
from torch.autograd.function import once_differentiable

#@once_differentiable
class renderFunctionSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cam_type:int,camera,points:torch.Tensor,environment_type:int,environment:torch.Tensor,ret_weights=False):
        #camera=list(map(float,camera))#Some components might be tensors and for some reason that breaks things
        ctx.do_camera_gradient=type(camera)==torch.Tensor
        
        plot,weights = kern.plotSinglePointsToTensor(cam_type,camera,points,environment_type,environment)
        
        ctx.save_for_backward(weights,torch.tensor(cam_type),torch.tensor(camera) if type(camera)==list else camera,points,plot,torch.tensor(environment_type),environment)
        if(args.depth_mode=='invert'):
            plot_inv=plot.clone()
            plot_inv[:,:,-1]=1/plot_inv[:,:,-1]
            plot=plot_inv
        elif(args.depth_mode=='remove'):
            plot=plot[:,:,:-1].clone()
        if(ret_weights):return plot,weights
        return plot
    
    @staticmethod
    def backward(ctx,plot_grad,weight_grad=None):
        weights,cam_type,camera,points,plot,environment_type,environment = ctx.saved_tensors
        if(args.depth_mode=='invert'):
            plot_grad[:,:,-1]=plot_grad[:,:,-1]/-(plot[:,:,-1]*plot[:,:,-1])
        elif(args.depth_mode=='remove'):
            plot_grad=torch.cat((plot_grad,torch.zeros_like(plot_grad[:,:,-1:])),-1)
        camera_grad,points_grad,environment_grad=kern.plotSinglePointsBackwardsToTensor(weights,int(cam_type),camera,points,int(environment_type),environment,plot,plot_grad.contiguous())
        return None,camera_grad if ctx.do_camera_gradient else None,points_grad,None,environment_grad
