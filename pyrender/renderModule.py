import torch
import kernelItf as kern

class renderFunctionSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cam_type:int,camera,points,environment):
        #camera=list(map(float,camera))#Some components might be tensors and for some reason that breaks things
        plot,weights = kern.plotSinglePointsToTensor(cam_type,camera,points,environment)
        ctx.save_for_backward([weights,camera,points,environment,plot])
        return plot
    @staticmethod
    def backward(ctx,plot_grad):
        weights,camera,points,environment,plot = ctx.saved_tensors
        camera_grad,points_grad,environment_grad=kern.plotSinglePointsBackwardsToTensor(weights,camera,points,environment,plot,plot_grad)
        return None,camera_grad,points_grad,environment_grad

#Now irrelevant?
class renderModule(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, cam_type:int,camera,points,environment):
        return renderFunctionSingle.apply(cam_type,camera,points,environment)
    
    pass
