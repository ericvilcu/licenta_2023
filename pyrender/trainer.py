import torch
import dataSet
from renderModule import renderFunctionSingle
import GLRenderer
from utility import spin_iter
from threading import Thread
import os

def error_test():
    #An error recommended I try this to check for some error.
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    data = torch.randn([1, 16, 32, 32], dtype=torch.float, device='cuda', requires_grad=True)
    net = torch.nn.Conv2d(16, 3, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1,device='cuda')
    net = net.cuda().float()
    out = net(data)
    out.backward(torch.randn_like(out).cuda())
    torch.cuda.synchronize()


error_test()




class GateModule(torch.nn.Module):
    #suggested to work well in most papers.
    #first described in: https://arxiv.org/pdf/1806.03589.pdf
    def __init__(self,in_channels,out_channels,kernel_size=(3,3),padding=1,single_gate=True,**kwargs):
        super().__init__()
        self.main=torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,**kwargs)
        self.register_module("main",self.main)
        self.gate=torch.nn.Conv2d(in_channels,1 if single_gate else out_channels,kernel_size=kernel_size,padding=padding,**kwargs)
        self.register_module("gate",self.gate)
    def forward(self,tsr:torch.Tensor)->torch.Tensor:
        return torch.nn.functional.elu(self.main(tsr))*torch.sigmoid(self.gate(tsr))


class MainModule(torch.nn.Module):
    
    def __init__(self,subplots=4,ndim=3,layers=None,kern=3,use_gates=True,inter_ch=16,empty=False,**unused_args):
        super().__init__()
        if(empty):return
        if(kern%2!=1):
            raise "kernel size must be odd for padding to be able to kee pit constant"
        self.norm = torch.nn.ELU() if not use_gates else torch.nn.Identity()
        if layers==None: layers= 1 if use_gates else 2
        self.register_module("norm", self.norm)
        padding=(kern-1)//2
        last_in=0
        self.in_layers=[];self.out_layers=[]
        for i in range(subplots):
            l=[]
            last_in+=ndim+1
            for j in range(layers):
                if(not use_gates):
                    mdl= torch.nn.Conv2d(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                else:
                    mdl=  GateModule(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                self.register_module(f"in{i}_{j}",mdl)
                l.append(mdl)
                last_in=inter_ch
            self.in_layers.append(l)
        last_in=0
        for i in range(subplots):
            l=[]
            last_in+=ndim+1+inter_ch
            for j in range(layers):
                if(not use_gates):
                    mdl= torch.nn.Conv2d(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                else:
                    mdl=  GateModule(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                self.register_module(f"out{i}_{j}",mdl)
                l.append(mdl)
                last_in=inter_ch
            self.out_layers.append(l)
            last_in=16
        self.final=torch.nn.Conv2d(last_in,3,kernel_size=(kern,kern),padding=padding)
        self.register_module("final", self.final)
        self.downsampler=torch.nn.AvgPool2d((2,2),stride=2)
        self.register_module("downsampler", self.downsampler)
    
    def required_subplots(self):
        return len(self.in_layers)
    
    def forward(self,imgs:list[torch.Tensor]):
        partials=[]
        prev = torch.zeros([0])
        for idx,layer in enumerate(self.in_layers):
            #needs to be transposed so that the functions work properly, as channels are expected to be the first dimension after batch in pytorch.
            img = imgs[idx].transpose(-3, -1).transpose(-2, -1)
            if (prev.size(0) != 0):
                img = torch.cat([img,prev], -3)
            for conv in layer:
                img = self.norm(conv(img))
            partials.append(img)
            prev = self.downsampler(img)
        prev = torch.zeros([0])
        
        for idx,layer in enumerate(self.out_layers):
            img = partials.pop()
            if (prev.size(0) != 0):
                if (len(prev.size()) == 3):
                    prev = torch.nn.functional.interpolate(torch.stack((prev,)),
                        mode='bilinear',align_corners=False,
                        size=[img.size(-2), img.size(-1) ])[0]
                else:
                    prev = torch.nn.functional.interpolate(prev,
                        mode='bilinear',align_corners=False,
                        size=[img.size(-2), img.size(-1) ])
                img = torch.cat((img,prev), -3)
            img = torch.cat([img,imgs[len(partials)].transpose(-3, -1).transpose(-2, -1)], -3)
            for conv in layer:
                img = self.norm(conv(img))
            prev = img

        prev = self.final(prev)
        #Now un-transpose it.
        return prev.transpose(-2,-1).transpose(-3,-1)


from torch.utils.data import DataLoader
from camera_utils import best_split_camera,downsize_camera,pad_camera,unpad_tensor,put_back_together,tensor_subsection
class trainer():
    def __init__(self,data:dataSet.DataSet,nn:MainModule=None,optim:torch.optim.Adam=None,**options) -> None:
        self.nn = MainModule(**options).cuda() if nn==None else nn.cuda()
        self.pad = int(options['start_padding']) if 'start_padding' in options else int(2**self.nn.required_subplots())
        self.optim = torch.optim.Adam([*self.nn.parameters(),*data.parameters()])if optim==None else optim
        self.data=data
        #TODO: split into train and validation. Maybe later.
        self.train_dataloader = DataLoader(self.data.trainImages, batch_size=1, shuffle=True,)
        self.MAX_PIXELS_PER_PLOT=int(1e9)
        self.test_dataloader = DataLoader(self.data.trainImages, batch_size=1, shuffle=False)
        self.test_dataloader_iter = spin_iter(self.test_dataloader)
        
        self.batch_size = int(options['batch_size']) if 'batch_size' in options else len(self.test_dataloader)
        self.report_loss=0.
        self.report_batches=0
    
    def save(self,path:str):
        if(not os.path.isdir(path)):
            if(os.path.exists(path)):
                raise Exception(f"Path {path} is invalid.")
            os.mkdir(path)
        data_path=os.path.join(path,"data")
        if(not os.path.isdir(data_path)):
            if(os.path.exists(data_path)):
                raise Exception(f"Path {data_path} is invalid.")
            os.mkdir(data_path)
        torch.save(self.nn,os.path.join(path,"model"))
        torch.save(self.optim,os.path.join(path,"optim"))
        
        self.data.save_to(data_path)
        

    @staticmethod
    def load(path:str,args:dict):
        data_path=os.path.join(path,"data")
        data=dataSet.DataSet.load(data_path)
        t=trainer(data=data,nn=torch.load(os.path.join(path,"model")),optim=torch.load(os.path.join(path,"optim")),**args,empty=True)
        return t
        
        
    def forward(self,plots:list[torch.Tensor]):
        return self.nn.forward(plots)
    
    def draw_subplots(self,scene_id:int,cam_type:int,cam_data:list[float],num:int):
        d=self.data.scenes[scene_id].data
        points,environment=d.points,d.environment
        plots:list[torch.Tensor]=[]
        for _unused in range(num):
            plots.append(renderFunctionSingle.apply(cam_type,cam_data,points,environment))
            cam_data=downsize_camera(cam_type,cam_data)
        return plots
    i=1000
    def train_diff(self,rez:torch.Tensor,target:torch.Tensor):
        rgb_target=target[::,::,:3:]
        alpha_target=target[::,::,-1::]
        if(trainer.i==0):
            trainer.i=1000
            print(*map(float,(rgb_target.mean(),rgb_target.max(),rgb_target.min())))
            print(*map(float,(rez.mean(),rez.max(),rez.min())))
        else:trainer.i-=1
        return torch.nn.functional.smooth_l1_loss(rez*alpha_target,rgb_target*alpha_target)
      
    def _train_one_unsafe(self,scene_id,cam_type,cameras:list[list[list[float]]],target):
        subplots=self.nn.required_subplots()
        total_diff=0.0
        for row in cameras:
            for cell in row:
                camera=cell
                plots=self.draw_subplots(scene_id,cam_type,pad_camera(camera,self.pad),subplots)
            
                rez=self.forward(plots)
                rez=unpad_tensor(rez,self.pad)
                
        
                diff=self.train_diff(rez,tensor_subsection(target,camera))
                total_diff+=float(diff)
                diff.backward()
        return total_diff

    
    def size_safe_forward(self,plots:list[torch.Tensor]):
        #TODO: implement
        pass
    
    def train_one_batch(self):
        try:
            diff=0
            for i in range(self.batch_size):
                scene_id,cam_type,camera,target,=next(iter(self.train_dataloader))
                target=torch.squeeze(target,0)
                cam_type=int(cam_type)
                W,H=camera[2],camera[3]
                if(W*H>self.MAX_PIXELS_PER_PLOT):
                    cams,W,H = best_split_camera(camera,self.MAX_PIXELS_PER_PLOT,expected_pad=self.pad)
                else:
                    cams=[[camera]]
                
                diff+=self._train_one_unsafe(scene_id,cam_type,cams,target)
                #TODO: also implement batches
            self.optim.step()
        except Exception as e:
            if(type(e)==torch.cuda.OutOfMemoryError):
                print("Memory error:",e,f"occurred, decreasing self.MAX_PIXELS_PER_PLOT to {W*H-1}; current batch has been skipped")
                self.MAX_PIXELS_PER_PLOT=int(W*H-1)
            elif(str(e)=="Unable to find a valid cuDNN algorithm to run convolution"):
                print("Weird cuDNN error:",e,f"occurred, assuming it's because of memory and decreasing self.MAX_PIXELS_PER_PLOT to {W*H-1}; current batch has been skipped")
                self.MAX_PIXELS_PER_PLOT=int(W*H-1)
            else:raise e
            if(self.MAX_PIXELS_PER_PLOT<4*self.pad*self.pad):raise Exception("MAX_PIXELS_PER_PLOT decreased too much. Your neural network might be too big to fit in your GPU memory")
        self.optim.zero_grad()
        self.report_loss+=diff
        self.report_batches+=1
    
    def example(self):
        try:
            scene_id,cam_type,camera,target,=next(iter(self.train_dataloader))
            W,H=camera[2],camera[3]
            pass
        except Exception as e:
            if(type(e)==torch.cuda.OutOfMemoryError):
                print("Memory error:",e,f"occurred, decreasing self.MAX_PIXELS_PER_PLOT to {W*H-1}")
                self.MAX_PIXELS_PER_PLOT=int(W*H-1)
            elif(str(e)=="Unable to find a valid cuDNN algorithm to run convolution"):
                print("Weird cuDNN error:",e,f"occurred, assuming it's because of memory and decreasing self.MAX_PIXELS_PER_PLOT to {W*H-1}")
                self.MAX_PIXELS_PER_PLOT=int(W*H-1)
            else:raise e
            if(self.MAX_PIXELS_PER_PLOT<self.pad):raise Exception("MAX_PIXELS_PER_PLOT decreased too much. Your nn might be too big to fit in your GPU memory")
        
    def display_results_to_renderer(self,r:GLRenderer.Renderer,points_view,target_view,result_view):
        scene_id,cam_type,camera,target,=self.test_dataloader_iter.next()
        target=torch.squeeze(target,0)
        cam_type=int(cam_type)
        #TODO: ensure plot does not crash due to image size.
        plots=self.draw_subplots(scene_id,cam_type,camera,self.nn.required_subplots())
        r.upload_tensor(points_view,plots[1])
        result = self.nn.forward(plots)
        r.upload_tensor(target_view,target)
        r.upload_tensor(result_view,result)
    
    def start_trainer_thread(self):
        self.tt = trainer_thread(self)
        self.tt.start()
        
    def stop_trainer_thread(self):
        self.tt.should_stop_training=True#not a data race due to GIL, but even if it was the consequences would not be disastrous.
        self.tt.join()
        del self.tt
import kernelItf
class trainer_thread(Thread):
    def __init__(self,parent) -> None:
        super().__init__()
        self.should_stop_training=False
        self.parent:trainer=parent
    def run(self):#TODO? reports on loss
        from time import time
        report_freq=30
        kernelItf.initialize()
        last_report=s=time()
        while(not self.should_stop_training):
            self.parent.train_one_batch()
            e=time()
            if(e-last_report>report_freq):
                print(f"Report: average loss is {self.parent.report_loss/self.parent.report_batches} in {self.parent.report_batches} batches")
                last_report=e
                
        kernelItf.cleanup()
