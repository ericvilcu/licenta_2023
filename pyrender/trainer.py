import torch
import dataSet
from renderModule import renderFunctionSingle
import GLRenderer
from utility import spin_iter
from threading import Thread
import math
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os
import args

psnr_module=PeakSignalNoiseRatio().cuda()
ssim_module=StructuralSimilarityIndexMeasure().cuda()
lpips_module=LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
vgg_module=LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
squeeze_module=LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
used_optim=torch.optim.Adam
LR_NN=1e-2
LR_DS=1e-2
LR_CAM=1e-5
TOTAL_BATCHES_THIS_RUN=0

def make_ch_first(x:torch.Tensor):
    return x.transpose(-3, -1).transpose(-2, -1)

def make_ch_last(x:torch.Tensor):
    return x.transpose(-2, -1).transpose(-3, -1)

def norm(x:torch.Tensor):
    return x.clamp(-1,1)
    
loss_functions={
    "l1":torch.nn.functional.l1_loss,
    "psnr":lambda src,target:psnr_module(make_ch_first(src),make_ch_first(target)),
    "ssim":lambda src,target:ssim_module(make_ch_first(src)[None,:],make_ch_first(target)[None,:]),
    "lpips_alex":lambda src,target:lpips_module(norm(make_ch_first(src)[None]),make_ch_first(target)[None]),
    "lpips_vgg":lambda src,target:vgg_module(norm(make_ch_first(src)[None]),make_ch_first(target)[None]),
    "lpips_squeeze":lambda src,target:squeeze_module(norm(make_ch_first(src)[None]),make_ch_first(target)[None]),
}

if args.main_loss!='':
    if args.main_loss in loss_functions:
        USED_LOSS=args.main_loss
        get_loss=lambda x:x[USED_LOSS]
    else:
        #a few special cases.
        if(args.main_loss=="l1+vgg"):
            USED_LOSS=args.main_loss
            get_loss=lambda x:x["lpips_vgg"]+x['l1']
        else:
            raise Exception(f"{args.main_loss} is not a known loss type. Available are {[*loss_functions]}(though ssim/psnr should never be used, and there are other special options like 'l1+vgg')")
else:
    USED_LOSS="lpips_vgg"
    get_loss=lambda x:x[USED_LOSS]

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


class GateConv(torch.nn.Module):
    #suggested to work well in a lot of point-based novel view synthesis papers.
    #as far as I know, it was first described in: https://arxiv.org/pdf/1806.03589.pdf
    def __init__(self,in_channels,out_channels,kernel_size=(3,3),padding=1,single_gate=True,main_norm=None,gate_norm=None,**kwargs):
        super().__init__()
        self.norm_main=torch.nn.ELU() if main_norm==None else main_norm
        self.norm_gate=torch.nn.Sigmoid() if gate_norm==None else gate_norm
        self.main=torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,**kwargs)
        self.gate=torch.nn.Conv2d(in_channels,1 if single_gate else out_channels,kernel_size=kernel_size,padding=padding,**kwargs)
    def forward(self,tsr:torch.Tensor)->torch.Tensor:
        return self.norm_main(self.main(tsr))*self.norm_gate(self.gate(tsr))

class GateModule(torch.nn.Module):
    #gate module, as described in https://arxiv.org/pdf/2205.05509.pdf
    def __init__(self,in_channels,out_channels=None,mid_channels=None,normalizer=None,kernel_size=(3,3),padding=1,**kwargs):
        super().__init__()
        if(out_channels==None):out_channels=in_channels
        if(mid_channels==None):mid_channels=in_channels
        self.normalizer=torch.nn.Identity() if normalizer==None else normalizer
        self.gate1=GateConv(in_channels,mid_channels,kernel_size=kernel_size,padding=padding,**kwargs)
        self.gate2=GateConv(mid_channels,mid_channels,kernel_size=(1,1),padding=0,**kwargs)
        self.gate3=GateConv(in_channels+mid_channels,out_channels,kernel_size=(1,1),padding=0,main_norm=torch.nn.Identity(),**kwargs)
        
    def forward(self,tsr:torch.Tensor)->torch.Tensor:
        return self.gate3(torch.cat([
            self.normalizer(self.gate2(self.normalizer(self.gate1(tsr))))
                                     ,tsr]))

# class SameScaleFusion(torch.nn.Module):
#     def __init__(self,in_channels,out_channels=None,mid_channels=None,normalizer=None,padding=1,single_gate=True,**kwargs):
#         super().__init__()
#     def forward(self,*tensors):
#         return sum(tensors)
    
# class MultiScaleFusion(torch.nn.Module):
#     def __init__(self,inputs,scale):
#         super().__init__()
#     def forward(self,*tensors):
#         return sum(tensors)

#A heavily simplified nn as described in READ. (I didn't figure out same-scale/multi-scale fusion yet.) uses their gate modules.
class MainModule_2(torch.nn.Module):
    def __init__(self,layers=1,subplots=4,ndim=3,empty=False,**unused_args):
        self.subplots=subplots
        channels=ndim+1
        super().__init__()
        if(empty):return
        padding=1
        kern=1+2*padding
        last_in=0
        self.in_layers=[];self.out_layers=[]
        for i in range(subplots):
            l=[]
            last_in+=channels
            for j in range(layers):
                mdl=GateConv(last_in,channels,kernel_size=(kern,kern),padding=padding)
                l.append(mdl)
            self.in_layers.append(torch.nn.ModuleList(l))
            last_in=channels
        last_in=0
        self.in_layers=torch.nn.ModuleList(torch.nn.ModuleList(layer) for layer in self.in_layers)
        for i in range(subplots):
            l=[]
            last_in+=channels+channels
            for j in range(layers):
                mdl=torch.nn.Sequential(
                    GateModule(last_in,channels,mid_channels=channels,kernel_size=(kern,kern),padding=padding),
                    GateConv(channels,channels,kernel_size=(kern,kern),padding=padding),
                )
                l.append(mdl)
            self.out_layers.append(torch.nn.ModuleList(l))
            last_in=channels
        self.out_layers=torch.nn.ModuleList(torch.nn.ModuleList(layer) for layer in self.out_layers)
        self.final=torch.nn.Conv2d(last_in,3,kernel_size=(1,1),padding=0)
        #self.register_module("final", self.final)
        self.downsampler=torch.nn.AvgPool2d((2,2),stride=2)
            
        
    def forward(self,imgs:list[torch.Tensor]):
        partials=[]
        prev = torch.zeros([0])
        for idx,layer in enumerate(self.in_layers):
            #needs to be transposed so that the functions work properly, as channels are expected to be the first dimension after batch in pytorch.
            img = imgs[idx].transpose(-3, -1).transpose(-2, -1)
            if (prev.size(0) != 0):
                img = torch.cat([img,prev], -3)
            for conv in layer:
                img = torch.nn.functional.elu(conv(img))
            partials.append(img)
            prev = self.downsampler(img)
        prev = torch.zeros([0])
        
        for idx,layer in enumerate(self.out_layers):
            img = partials.pop()
            if (prev.size(0) != 0):
                if (len(prev.size()) == 3):
                    prev = torch.nn.functional.interpolate(torch.stack((prev,)),
                        mode='bilinear',align_corners=True,
                        size=[img.size(-2), img.size(-1) ])[0]
                else:
                    prev = torch.nn.functional.interpolate(prev,
                        mode='bilinear',align_corners=True,
                        size=[img.size(-2), img.size(-1) ])
                img = torch.cat((img,prev), -3)
            img = torch.cat([img,imgs[len(partials)].transpose(-3, -1).transpose(-2, -1)], -3)
            for conv in layer:
                img = conv(img)
            prev = img

        #print([*self.final.parameters()][0][0][0],[*self.final.parameters()][1])
        prev = self.final(img)
        #Now un-transpose it.
        return prev.transpose(-2,-1).transpose(-3,-1)

    def required_subplots(self):
        return self.subplots

class MainModule_1(torch.nn.Module):
    
    def __init__(self,subplots=4,ndim=3,layers=None,kern=3,use_gates=True,inter_ch=16,empty=False,**unused_args):
        super().__init__()
        if(empty):return
        if(kern%2!=1):
            raise "kernel size must be odd for padding to be able to keep it constant"
        self.norm = torch.nn.ELU() if not use_gates else torch.nn.Identity()
        if layers==None: layers= 1 if use_gates else 2
        #self.register_module("norm", self.norm)
        padding=(kern-1)//2
        last_in=0
        self.in_layers=[];self.out_layers=[]
        for i in range(subplots):
            l=[]
            last_in+=ndim+1
            for j in range(layers):
                if(not use_gates):
                    mdl=torch.nn.Conv2d(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                else:
                    mdl=     GateConv(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                #self.register_module(f"in{i}_{j}",mdl)
                l.append(mdl)
                last_in=inter_ch
            self.in_layers.append(l)
        last_in=0
        self.in_layers=torch.nn.ModuleList(torch.nn.ModuleList(layer) for layer in self.in_layers)
        for i in range(subplots):
            l=[]
            last_in+=ndim+1+inter_ch
            for j in range(layers):
                if(not use_gates):
                    mdl=torch.nn.Conv2d(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                else:
                    mdl=     GateConv(last_in,inter_ch,kernel_size=(kern,kern),padding=padding)
                #self.register_module(f"out{i}_{j}",mdl)
                l.append(mdl)
                last_in=inter_ch
            self.out_layers.append(l)
            last_in=16
        self.out_layers=torch.nn.ModuleList(torch.nn.ModuleList(layer) for layer in self.out_layers)
        self.final=torch.nn.Conv2d(last_in,3,kernel_size=(1,1),padding=0)
        #self.register_module("final", self.final)
        self.downsampler=torch.nn.AvgPool2d((2,2),stride=2)
        #self.register_module("downsampler", self.downsampler)
    
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
                        mode='bilinear',align_corners=True,
                        size=[img.size(-2), img.size(-1) ])[0]
                else:
                    prev = torch.nn.functional.interpolate(prev,
                        mode='bilinear',align_corners=True,
                        size=[img.size(-2), img.size(-1) ])
                img = torch.cat((img,prev), -3)
            img = torch.cat([img,imgs[len(partials)].transpose(-3, -1).transpose(-2, -1)], -3)
            for conv in layer:
                img = self.norm(conv(img))
            prev = img

        #print([*self.final.parameters()][0][0][0],[*self.final.parameters()][1])
        prev = self.final(img)
        #Now un-transpose it.
        return prev.transpose(-2,-1).transpose(-3,-1)


def build_main_module(**options):
    if(args.nn_type==1):
        return MainModule_1(**options)
    elif(args.nn_type==2):
        return MainModule_2(**options)
from torch.utils.data import DataLoader
from camera_utils import best_split_camera,downsize_camera,pad_camera,unpad_tensor,put_back_together,tensor_subsection
class trainer():
    def __init__(self,data:dataSet.DataSet,nn=None,optim:torch.optim.Adam=None,default_path=None,**options) -> None:
        self.nn = (build_main_module(**options).cuda() if nn==None else nn.cuda()).requires_grad_(args.nn_refinement)
        self.pad = int(options['start_padding']) if 'start_padding' in options else int(2**self.nn.required_subplots())
        self.default_path=default_path
        self.optim=used_optim([
            {'params':self.nn.parameters()},{'params':data.parameters(),'lr':LR_DS}
        ],lr=LR_NN) if optim==None else optim
        self.data=data
        #TODO: split into train and validation. Maybe later.
        self.train_dataloader = DataLoader(self.data.trainImages, batch_size=1, shuffle=True,)
        self.train_dataloader_iter = spin_iter(self.train_dataloader)
        self.MAX_PIXELS_PER_PLOT=int(1e9)
        self.test_dataloader = DataLoader(self.data.trainImages, batch_size=1, shuffle=False)
        self.test_dataloader_iter = spin_iter(self.test_dataloader)
        
        self.batch_size = int(options['batch_size']) if 'batch_size' in options else len(self.test_dataloader)
        self.report_losses={}
        self.report_batches=0
    
    def save(self,path:str=None):
        if(path==None):
            path=self.default_path
        if(path==None):
            raise Exception("Please provide a path if the nn was not loaded")
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
        torch.save({"optim":self.optim.state_dict()},os.path.join(path,"optim"))
        
        self.data.save_to(data_path)
        

    @staticmethod
    def load(path:str,args:dict):
        data_path=os.path.join(path,"data")
        data=dataSet.DataSet.load(data_path)
        nn=torch.load(os.path.join(path,"model"))
        optim=used_optim([
            {'params':nn.parameters()},{'params':data.parameters(),'lr':LR_DS}
        ],lr=LR_NN)
        optim.load_state_dict(torch.load(os.path.join(path,"optim"))['optim'])
        optim.zero_grad()
        t=trainer(data=data,nn=nn,optim=optim,**args,empty=True,default_path=path)
        return t
        
        
    def forward(self,plots:list[torch.Tensor]):
        return self.nn.forward(plots)
    
    def draw_subplots(self,scene_id:int,cam_type:int,cam_data:list[float] or torch.Tensor,num:int):
        d=self.data.scenes[scene_id].data
        points,environment,environment_type=d.points,d.environment,d.environment_type
        plots:list[torch.Tensor]=[]
        for _unused in range(num):
            plots.append(renderFunctionSingle.apply(cam_type,cam_data,points,environment_type,environment))
            cam_data=downsize_camera(cam_type,cam_data)
        return plots
    def train_diff(self,rez:torch.Tensor,target:torch.Tensor,mul:float):
        rgb_target=target[::,::,:3:]
        alpha_target=target[::,::,-1::]
        real_src,real_target=rez*alpha_target,rgb_target*alpha_target
        return {i:mul*loss_functions[i](real_src,real_target) for i in loss_functions}
      
    def _train_one_unsafe(self,scene_id,cam_type,cameras:list[list[list[float]]],target):
        subplots=self.nn.required_subplots()
        total_diff={name:0. for name in loss_functions}
        for row in cameras:
            for cell in row:
                camera=cell
                plots=self.draw_subplots(scene_id,cam_type,pad_camera(camera,self.pad),subplots)
            
                rez=self.forward(plots)
                rez=unpad_tensor(rez,self.pad)*camera[4]
                
        
                diff=self.train_diff(rez,tensor_subsection(target,camera),1/(len(cameras)*len(row)*self.batch_size))
                for name in loss_functions:
                    total_diff[name]+=float(diff[name])
                global get_loss
                get_loss(diff).backward()
                diff={}
        return total_diff

    def size_safe_forward_nograd(self,cam_type:int,camera_data:list[float],scene:int):
        with torch.no_grad():
            #TODO: image split if necessary
            lum=camera_data[4]
            subplots=self.draw_subplots(scene,cam_type,pad_camera(camera_data,self.pad),self.nn.required_subplots())
            img_float=unpad_tensor(self.forward(subplots),self.pad)
            if(lum<=0):
                lum=2/img_float.mean()
            return img_float*lum
        
    
    def train_one_batch(self):
        try:
            diff={name:0 for name in loss_functions}
            for i in range(self.batch_size):
                scene_id,cam_type,camera,target,updater=next(iter(self.train_dataloader_iter))
                camera=camera[0]
                camera.requires_grad=args.improve_cameras
                target=torch.squeeze(target,0)
                cam_type=int(cam_type)
                W,H=camera[2],camera[3]
                if(W*H>self.MAX_PIXELS_PER_PLOT):
                    cams,W,H = best_split_camera(camera,self.MAX_PIXELS_PER_PLOT,expected_pad=self.pad)
                else:
                    cams=[[camera]]
                
                local_diff=self._train_one_unsafe(scene_id,cam_type,cams,target)
                for name in loss_functions:
                    diff[name]+=float(local_diff[name])
                if(args.camera_refinement and type(camera)==torch.Tensor):
                    if(camera.grad!=None):
                        #TODO: get rid of magic number, somehow.
                        dataSet.camera_updater(**updater)(camera+camera.grad*LR_CAM)
                local_diff={}
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
        for name in diff:
            if name in self.report_losses:
                self.report_losses[name]+=diff[name]
            else:
                self.report_losses[name]=diff[name]
        self.report_batches+=1
        global TOTAL_BATCHES_THIS_RUN
        TOTAL_BATCHES_THIS_RUN+=1
        return get_loss(diff)
        
    def display_results_to_renderer(self,r:GLRenderer.Renderer,points_view,target_view,result_view):
        scene_id,cam_type,camera,target,*unused=self.test_dataloader_iter.next()
        target=torch.squeeze(target,0)
        cam_type=int(cam_type)
        #TODO: ensure plot does not crash due to image size.
        plots=self.draw_subplots(scene_id,cam_type,camera[0] if camera.size(0)==1 else camera,self.nn.required_subplots())
        r.upload_tensor(points_view,plots[0])
        result = self.nn.forward(plots)
        r.upload_tensor(target_view,target)
        r.upload_tensor(result_view,result)
    
    def start_trainer_thread(self):
        assert('tt' not in dir(self))
        self.tt = trainer_thread(self)
        self.tt.start()
        
    def stop_trainer_thread(self):
        self.tt.should_stop_training=True#not a data race due to GIL, but even if it was the consequences would not be disastrous.
        self.tt.join()
        del self.tt

    #image saving stuff
    def save_one_(self):
        self.save_one(int(0.33*len(self.data.trainImages)))
    def save_one(self,id,title=None):
        from torchvision.utils import save_image
        scene_id,cam_type,camera,*unused=self.data.trainImages.__getitem__(id)
        c=self.size_safe_forward_nograd(cam_type,camera,scene_id)
        if(title==None):
            title=f"i{id}_b{TOTAL_BATCHES_THIS_RUN}"
        save_image(c.transpose(-3,-1).transpose(-2,-1),os.path.join(args.sample_folder,args.sample_prefix+title+".png"))
    def save_all_samples(self):
        for i in range(len(self.data.trainImages)):
            self.save_one(i,title=f"final_{i}")

if(args.stagnation_batches!=-1):
    ln=[1e20]*args.stagnation_batches
    cn=[1e10]*args.stagnation_batches
    stagnated:bool=False
    def append_loss_history(l:float):
        global ln,cn,stagnated
        ln=ln[1:]+[cn[0]]
        cn=cn[1:]+[l]
        stagnated=sum(cn)*(1+args.stagnation_p)>sum(ln)
        if(stagnated):
            print("NN improvement seems to have stagnated...")
    def is_stagnant() -> bool:
        return stagnated
import kernelItf
class trainer_thread(Thread):
    def __init__(self,parent,report_freq=120) -> None:
        super().__init__()
        self.should_stop_training=False
        self.report_freq=report_freq
        self.parent:trainer=parent
    def run(self):
        from time import time
        self.parent.nn.train()
        self.parent.data.train()
        kernelItf.initialize()
        last_save=last_report=s=time()
        while(not self.should_stop_training):
            if(args.samples_every>0 and (TOTAL_BATCHES_THIS_RUN%args.samples_every==0)):
                self.parent.save_one_()
            loss_this_batch=self.parent.train_one_batch()
            e=time()
            if(e-last_report>self.report_freq or self.should_stop_training and self.parent.report_batches>0):
                l={loss:float(self.parent.report_losses[loss]/self.parent.report_batches) for loss in self.parent.report_losses}
                print(f"Report: average loss is {l} in {self.parent.report_batches} batches")
                dt=e-s
                print(f"Total batches:{TOTAL_BATCHES_THIS_RUN};Runtime={int(dt)//3600:02d}:{(int(dt)//60)%60:02d}:{int(dt)%60:02d}.{str(math.fmod(dt,1))[2:]}")
                l={}
                last_report=e
                self.parent.report_losses={}
                self.parent.report_batches=0
            if(e-last_save>args.autosave_s):
                tmp=time()
                print("autosaving...")
                last_save=time()
                print(f"autosaved ({last_save-tmp}s)")
                self.parent.save()
            if(args.stagnation_batches!=-1):
                append_loss_history(loss_this_batch)
        kernelItf.cleanup()
