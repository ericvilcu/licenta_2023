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
from torchvision.utils import save_image
import os
import args
import time

psnr_module=PeakSignalNoiseRatio().cuda()
ssim_module=StructuralSimilarityIndexMeasure().cuda()
lpips_module=LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
vgg_module=LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
squeeze_module=LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()
used_optim=torch.optim.Adam
LR_NN=1e-2#TODO: arguments for these 3; for now i'll just set them manually
LR_DS=1e-3
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
        elif(not args.main_loss.startswith("i")):
            raise Exception(f"{args.main_loss} is not a known loss type. Available are {[*loss_functions]}(though ssim/psnr should never be used, and there are other special options like 'l1+vgg')")
        else:
            #DANGEROUS!
            get_loss=lambda x:eval(args.main_loss[1:],x.copy(),x.copy())
else:
    USED_LOSS="lpips_vgg"
    get_loss=lambda x:x[USED_LOSS]

def error_test():
    #An error message recommended I try this to check for some error.
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


class trainMetadata():
    def __init__(self,path:str=None,hist:list=None,batches:int=None,run_hist=None,times=None):
        self.hist=[]
        self.times=[]
        self.timesValidation=[]
        self.histValidation=[]
        self.batches=0
        self.run_hist={0:({i:getattr(args.raw_args,i) for i in dir(args.raw_args) if not i.startswith('_')},time.time())}
        if(path!=None):
            if(os.path.exists(path)):
                fo=torch.load(path)
                self.hist=fo['hist']
                self.batches=fo['batches']
                true_run_hist={**fo['run_hist']} if 'run_hist' in fo else self.run_hist
                true_run_hist[self.batches]=self.run_hist[0]
                self.run_hist=true_run_hist
                self.times=fo['times']
                self.timesValidation=fo['timesValidation']
                self.histValidation=fo['histValidation']
        elif(hist!=None and batches!=None):
            self.hist=hist
            self.batches=batches
            self.run_hist=run_hist if run_hist!=None else self.run_hist
            self.times=times if times!=None else [-1 for i in self.hist]
        self.my_run=self.batches
    def saveable(self):
        return {'hist':self.hist,'batches':self.batches,'run_hist':self.run_hist,'times':self.times,'timesValidation':self.timesValidation,'histValidation':self.histValidation}

metaData=trainMetadata()
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
        channels=ndim+(args.depth_mode!='remove')
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

class MainModule_3(torch.nn.Module):
    #NOTE: implementation is lazy
    def __init__(self, subplots=4, ndim=5, empty=False, connectivity_mode=0, **unused_args) -> None:
        if(args.depth_mode!='invert'):
            print("WARN: this type of nn works poorly without \"--depth_mode invert\"  as arguments.")
        if(args.depth_mode=='remove'):
            raise Exception("This type of nn requires depth.")
        self.layers=subplots
        super().__init__()
        self.dir_map={0:[3,4,5,6],1:[3,3,4,4],2:[3,3,3,3]}[connectivity_mode]
        if(ndim!=3+len(set(self.dir_map))):
            ex=len(set(self.dir_map))
            raise Exception(f"ndim must be equal to {3+ex} ({ex} extra channels) (rgb,{'x+/x-/y+/y-' if ex==4 else('x/y' if ex==2 else 'overall')} connectivity, depth/weight) for connectivity mode {connectivity_mode}")
        self.kp=torch.nn.Parameter(torch.tensor(1.).cuda())
        self.pp=torch.nn.Parameter(torch.tensor(1./12.).cuda())
        self.dp=torch.nn.Parameter(torch.tensor(1./4.).cuda())
    
    def colors(self,tsr):
        return tsr[:,:,:-1]
    def weight(self,tsr):
        return tsr[:,:,-1:]
    
    def weighted_sum(self,*plots:torch.Tensor):
        plots=list(plots)
        for i in range(len(plots)-1):
            if(plots[i].size()!=plots[-1].size()):
                plots[i]=torch.nn.functional.interpolate(torch.stack((torch.stack((plots[i],)),)),#welcome to stupid.
                            mode='nearest',
                            size=[plots[-1].size(-3), plots[-1].size(-2), plots[-1].size(-1) ])[0][0]
        plots_w=[self.weight(plot) for plot in plots]
        plots_c=[self.colors(plot) for plot in plots]
        weight_sum: torch.Tensor=sum(plots_w)
        clr=sum([w*c for w,c in zip(plots_w,plots_c)])/weight_sum
        assert((weight_sum<=0).any()==False)
        return torch.cat([clr,torch.log(1+weight_sum)],-1)#todo: replace log? it is a little out of place.
    
    def apply_connectivity(self,plot):
        #return plot
        #xn_yn=plot
        z=torch.ones_like if True else torch.nn.Identity()
        xny0=torch.cat([z(plot[:,-1:]),  plot[:,:-1] ],1)
        xny1=torch.cat([  plot[:,1:]  ,z(plot[:,:1] )],1)
        x0yn=torch.cat([z(plot[-1:]  ),  plot[:-1]   ],0)
        x1yn=torch.cat([  plot[1:]    ,z(plot[:1]   )],0)
        x0y0=torch.cat([z(xny0[-1:]  ),  xny0[:-1]   ],0)
        x1y0=torch.cat([  xny0[1:]    ,z(xny0[:1]   )],0)
        x0y1=torch.cat([  x0yn[:,1:]  ,z(x0yn[:,:1] )],1)
        x1y1=torch.cat([  x1yn[:,1:]  ,z(x1yn[:,:1] )],1)
        xy_mat=[[x0y0,xny0,x1y0],
                      [x0yn,plot,x1yn],
                      [x0y1,xny1,x1y1],]
        results=[torch.cat([plot,torch.ones_like(plot[:,:,-1:])],-1)]
        
        s2=1/2#math.sqrt(2)
        d1=(           1/2)/(math.sqrt(3)/2 + 1/2) * self.kp
        d2=(math.sqrt(3)/2)/(math.sqrt(3)/2 + 1/2) * self.kp
        s4=self.dp
        for (I1,J1,R1),(I2,J2,R2) in [
            # 2 main crosses. X and +
            ((0,0,((0,s2),(2,s2))),(2,2,((1,s2),(3,s2)))),#\
            ((0,2,((0,s2),(3,s2))),(2,0,((1,s2),(2,s2)))),#/
            ((1,0,((0, 1),      )),(1,2,((1, 1),      ))),#|
            ((0,1,((2, 1),      )),(2,1,((3, 1),      ))),#-
            
            #Then theres also 8 knight moves. The way they're written is a bit of a mess...
            ((1,0,((1,d2),(2,d1))),(0,2,((0,d1),(3,d1)))),#A 1 /
            ((1,0,((1,d2),(3,d1))),(2,2,((0,d1),(2,d1)))),#A 2 \
                
            ((0,0,((1,d2),(3,d1))),(1,2,((0,d2),(2,d1)))),#V 1 \
            ((2,0,((1,d2),(2,d1))),(1,2,((0,d2),(3,d1)))),#V 2 /
                
            ((2,0,((1,d1),(2,d2))),(0,1,((0,d1),(3,d2)))),#< 1 /
            ((0,1,((1,d1),(3,d2))),(2,2,((0,d1),(2,d2)))),#< 2 \
                
            ((0,0,((1,d1),(3,d2))),(2,1,((0,d1),(2,d2)))),#> 1 \
            ((2,1,((1,d1),(2,d2))),(0,2,((0,d1),(3,d2)))),#> 2 /
            
            #We can also consider the little diamond.
            ((1,0,((0,s4),(2,s4))),(2,1,((1,s4),(3,s4)))),#\ left
            ((1,0,((0,s4),(3,s4))),(1,0,((1,s4),(2,s4)))),#/ left
            ((0,1,((0,s4),(2,s4))),(1,2,((1,s4),(3,s4)))),#\ right
            ((1,2,((0,s4),(3,s4))),(2,1,((1,s4),(2,s4)))),#/ right
        ]:
            R1=[(self.dir_map[i],2) for i,w in R1]
            R2=[(self.dir_map[i],2) for i,w in R2]
            # w1=self.weight(x0y0)#xy_mat[I1-1][J1-1])#counted from 1 on accident. I blame octave.
            # w2=self.weight(x1y1)#xy_mat[I2-1][J2-1])
            # c1=self.colors(x0y0)#xy_mat[I1-1][J1-1])
            # c2=self.colors(x1y1)#xy_mat[I2-1][J2-1])
            w1=self.weight(xy_mat[I1][J1])
            w2=self.weight(xy_mat[I2][J2])
            c1=self.colors(xy_mat[I1][J1])
            c2=self.colors(xy_mat[I2][J2])
            #The -1 is b/c I counted from 1 on accident. I blame octave.
            relu2 = lambda x:1+torch.nn.functional.elu(x)
            factor=relu2(sum([c1[:,:,R-1:R]*w for R,w in R1]))*relu2(sum([c2[:,:,R-1:R]*w for R,w in R2]))
            factor2=((w1+w2)*factor)
            if (factor2==-torch.inf).any() or (factor2==torch.inf).any() or (factor2.isnan()).any():
                raise Exception("OH NO")
            if (w1+w2<=0).any():
                raise Exception("OH NO")
            results.append(
                torch.cat([(c1*w1+c2*w2)/(w1+w2),(w1+w2),(w1+w2)*factor],-1)
            )
        ret=self.weighted_sum(*results)[::,::,:-1]
        #print(*[list(map(float,(r[:,:,:3].max(),r[:,:,:3].min(),r[:,:,-1].max(),r[:,:,-1].min()))) for r in [plot,ret]])
        return ret
    
    def forward(self,plots:list[torch.Tensor]):
        part_rez=None
        #start from the lowest rez.
        for plot in plots[::-1]:
            #plot=self.apply_connectivity(plot)
            #plot[:,:,-1]=(2*torch.ones_like(plot[:,:,-1])).pow(6*plot[:,:,-1])
            plot=torch.cat([plot[:,:,:-1],torch.sqrt(plot[:,:,-1:])],-1)
            if(part_rez!=None):
                part_rez=torch.cat([part_rez[:,:,:-1],part_rez[:,:,-1:]*self.pp],-1)
                part_rez=self.weighted_sum(part_rez,plot)
            else:
                part_rez=plot
            part_rez=self.apply_connectivity(part_rez)
        return part_rez[:,:,:3]
    
    def required_subplots(self):
        return self.layers
    
    
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
        ndim_d=ndim+(args.depth_mode!='remove')
        for i in range(subplots):
            l=[]
            last_in+=ndim_d
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
            last_in+=ndim_d+inter_ch
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
    elif(args.nn_type==3):
        return MainModule_3(**options)
    else:
        raise Exception("Invalid module type index")

from torch.utils.data import DataLoader
from camera_utils import best_split_camera,downsize_camera,pad_camera,unpad_tensor,put_back_together,tensor_subsection
class trainer():
    def __init__(self,data:dataSet.DataSet,nn=None,optim:torch.optim.Adam=None,default_path=None,**options) -> None:
        self.nn = (build_main_module(**options).cuda() if nn==None else nn.cuda()).requires_grad_(args.nn_refinement)
        self.pad = int(options['start_padding']) if 'start_padding' in options else int(2**(self.nn.required_subplots()))
        self.default_path=default_path
        self.optim=used_optim([
            {'params':data.parameters(),'lr':LR_DS,'name':'data'},{'params':self.nn.parameters(),'name':'nn'}
        ],lr=LR_NN) if optim==None else optim
        self.data=data
        
        self.train_dataloader = DataLoader(dataSet.SubDataSet(self.data.trainImages,10,2,False), batch_size=1, shuffle=True)
        self.train_dataloader_iter = spin_iter(self.train_dataloader)
        
        self.validation_dataloader  = DataLoader(dataSet.SubDataSet(self.data.trainImages,10,2,True), batch_size=1, shuffle=False)
        self.validation_iterator = spin_iter(self.validation_dataloader)
        
        self.test_dataloader  = DataLoader(self.data.trainImages, batch_size=1, shuffle=False)
        self.test_dataloader_iter = spin_iter(self.test_dataloader)
        
        self.MAX_PIXELS_PER_PLOT_NOGRAD=self.MAX_PIXELS_PER_PLOT=int(1e9)
        
        self.batch_size = int(options['batch_size']) if 'batch_size' in options else len(self.train_dataloader)
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
        global metaData
        torch.save(metaData.saveable(),os.path.join(path,"meta"))
        
        self.data.save_to(data_path)
        

    @staticmethod
    def load(path:str,args:dict):
        global metaData
        data_path=os.path.join(path,"data")
        data=dataSet.DataSet.load(data_path)
        metaData=trainMetadata(path=os.path.join(path,"meta"))
        nn=torch.load(os.path.join(path,"model"))
        optim=used_optim([
            {'params':data.parameters(),'lr':LR_DS,'name':'data'},{'params':nn.parameters(),'name':'nn'}
        ],lr=LR_NN)
        optim.load_state_dict(torch.load(os.path.join(path,"optim"))['optim'])
        optim.param_groups[0]['lr']=LR_NN if optim.param_groups[0]['name']=='nn' else LR_DS
        optim.param_groups[1]['lr']=LR_NN if optim.param_groups[1]['name']=='nn' else LR_DS
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
            plots.append(renderFunctionSingle.apply(cam_type,cam_data,points,environment_type,d.get_environment()))
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
                #torch.cuda.synchronize()
                rez=self.forward(plots)
                #torch.cuda.synchronize()
                rez=unpad_tensor(rez,self.pad)*camera[4]
                #torch.cuda.synchronize()
                
        
                diff=self.train_diff(rez,tensor_subsection(target,camera),1/(len(cameras)*len(row)*self.batch_size))
                for name in loss_functions:
                    total_diff[name]+=float(diff[name])
                global get_loss
                assert(get_loss(diff)>=0)
                get_loss(diff).backward()
                diff={}
        return total_diff

    def size_safe_forward_nograd(self,cam_type:int=None,camera:list[float]=None,scene:int=None):
        try:
            with torch.no_grad():
                lum=camera[4]
                
                W,H=camera[2],camera[3]
                if(W*H>self.MAX_PIXELS_PER_PLOT_NOGRAD):
                    cams,W,H = best_split_camera(camera,self.MAX_PIXELS_PER_PLOT_NOGRAD,expected_pad=self.pad)
                else:
                    cams=[[camera]]
                plots=[[self.nn.forward(self.draw_subplots(scene,cam_type,pad_camera(cam,self.pad),self.nn.required_subplots()))
                        for cam in cam_col] for cam_col in cams]
                plots=put_back_together(plots,self.pad)
                if(lum<=0):
                    lum=1/plots.mean()
                return plots*lum
        except Exception as e:
            if(type(e)==torch.cuda.OutOfMemoryError):
                print("Memory error:",e,f"occurred, decreasing self.MAX_PIXELS_PER_PLOT_NOGRAD to {W*H-1}; current batch has been skipped")
                self.MAX_PIXELS_PER_PLOT_NOGRAD=int(W*H-1)
            elif(str(e)=="Unable to find a valid cuDNN algorithm to run convolution"):
                print("Weird cuDNN error:",e,f"occurred, assuming it's because of memory and decreasing self.MAX_PIXELS_PER_PLOT_NOGRAD to {W*H-1}; current batch has been skipped")
                self.MAX_PIXELS_PER_PLOT_NOGRAD=int(W*H-1)
            else:raise e
            if(self.MAX_PIXELS_PER_PLOT_NOGRAD<4*self.pad*self.pad):raise Exception("MAX_PIXELS_PER_PLOT_NOGRAD decreased too much. Your neural network might be too big to fit in your GPU memory")
            lum=None;cams=None;plots=None;W=H=None
            return self.size_safe_forward_nograd(cam_type,camera,scene)
        
    
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
                        dataSet.camera_updater(**updater)(camera,camera.grad*LR_CAM)
                local_diff={}
            metaData.batches+=1
            global TOTAL_BATCHES_THIS_RUN
            TOTAL_BATCHES_THIS_RUN+=1
            metaData.times.append(time.time())
            metaData.hist.append(diff)
            self.optim.step()
            for name in diff:
                if name in self.report_losses:
                    self.report_losses[name]+=diff[name]
                else:
                    self.report_losses[name]=diff[name]
            self.report_batches+=1
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
        return get_loss(diff)
    
    def _validate_one_unsafe(self,scene_id,cam_type,cameras:list[list[list[float]]],target):
        subplots=self.nn.required_subplots()
        total_diff={name:0. for name in loss_functions}
        for row in cameras:
            for cell in row:
                if(args.camera_refinement):
                    camera=cell
                    plots=self.draw_subplots(scene_id,cam_type,pad_camera(camera,self.pad),subplots)
                    rez=self.forward(plots)
                    rez=unpad_tensor(rez,self.pad)*camera[4]
                    diff=self.train_diff(rez,tensor_subsection(target,camera),1/(len(cameras)*len(row)*len(self.validation_iterator)))
                    for name in loss_functions:
                        total_diff[name]+=float(diff[name])
                    global get_loss
                    get_loss(diff).backward()
                    diff={}
                else:
                    with torch.no_grad():
                        camera=cell
                        plots=self.draw_subplots(scene_id,cam_type,pad_camera(camera,self.pad),subplots)
                        rez=self.forward(plots)
                        rez=unpad_tensor(rez,self.pad)*camera[4]
                        diff=self.train_diff(rez,tensor_subsection(target,camera),1/(len(cameras)*len(row)*len(self.validation_iterator)))
                        for name in loss_functions:
                            total_diff[name]+=float(diff[name])
                        #global get_loss
                        #get_loss(diff).backward()
                        diff={}
                    
        return total_diff
    
    def validate_one_batch(self):
        try:
            diff={name:0 for name in loss_functions}
            for i in range(len(self.validation_iterator)):
                scene_id,cam_type,camera,target,updater=next(iter(self.validation_iterator))
                camera=camera[0]
                camera.requires_grad=False#NOTE: I need to do something about that. Camera position/luminosity not being refined may be a problem.
                target=torch.squeeze(target,0)
                cam_type=int(cam_type)
                W,H=camera[2],camera[3]
                if(W*H>self.MAX_PIXELS_PER_PLOT):
                    cams,W,H = best_split_camera(camera,self.MAX_PIXELS_PER_PLOT,expected_pad=self.pad)
                else:
                    cams=[[camera]]
                
                local_diff=self._validate_one_unsafe(scene_id,cam_type,cams,target)
                for name in loss_functions:
                    diff[name]+=float(local_diff[name])

                if(args.camera_refinement and type(camera)==torch.Tensor):
                    if(camera.grad!=None):
                        #TODO: get rid of magic number, somehow.
                        dataSet.camera_updater(**updater)(camera,camera.grad*LR_CAM)
                
                local_diff={}
            metaData.timesValidation.append(time.time())
            metaData.histValidation.append(diff)
            self.optim.zero_grad()
        except Exception as e:
            if(type(e)==torch.cuda.OutOfMemoryError):
                print("Memory error:",e,f"occurred, decreasing self.MAX_PIXELS_PER_PLOT to {W*H-1}; current batch has been skipped")
                self.MAX_PIXELS_PER_PLOT=int(W*H-1)
            elif(str(e)=="Unable to find a valid cuDNN algorithm to run convolution"):
                print("Weird cuDNN error:",e,f"occurred, assuming it's because of memory and decreasing self.MAX_PIXELS_PER_PLOT to {W*H-1}; current batch has been skipped")
                self.MAX_PIXELS_PER_PLOT=int(W*H-1)
            else:raise e
            if(self.MAX_PIXELS_PER_PLOT<4*self.pad*self.pad):raise Exception("MAX_PIXELS_PER_PLOT decreased too much. Your neural network might be too big to fit in your GPU memory")
        return {**diff,'get_loss':get_loss(diff)}
        
    def display_results_to_renderer(self,r:GLRenderer.Renderer,points_view,result_view,target_view):
        with torch.no_grad():
            scene_id,cam_type,camera,target,*unused=self.test_dataloader_iter.next()
            target=torch.squeeze(target,0)
            cam_type=int(cam_type)
            camera=camera[0] if camera.size(0)==1 else camera
            #TODO: ensure plot does not crash due to image size.
            r.upload_tensor(target_view,target)
            target=None
            plots=self.draw_subplots(scene_id,cam_type,camera,1)
            r.upload_tensor(points_view,plots[0])
            plots=None
            result = self.size_safe_forward_nograd(cam_type=cam_type,camera=camera,scene=scene_id)
            r.upload_tensor(result_view,result)
    
    def start_trainer_thread(self,cond):
        assert('tt' not in dir(self))
        self.tt = trainer_thread(self,cond)
        self.tt.start()
        
    def stop_trainer_thread(self):
        self.tt.should_stop_training=True#not a data race due to GIL, but even if it was the consequences would not be disastrous.
        self.tt.join()
        del self.tt

    #image saving stuff
    def save_one_(self):
        self.save_one(int(0.33*len(self.data.trainImages)))
    def save_one(self,id,title=None):
        scene_id,cam_type,camera,*unused=self.data.trainImages.__getitem__(id)
        c=self.size_safe_forward_nograd(cam_type,camera,scene_id)
        if(title==None):
            title=f"i{id}_b{metaData.batches}"
        if(not os.path.exists(args.sample_folder)):os.mkdir(args.sample_folder)
        save_image(c.transpose(-3,-1).transpose(-2,-1),os.path.join(args.sample_folder,args.sample_prefix+title+".png"))
    def save_all_samples(self):
        for i in range(len(self.data.trainImages)):
            self.save_one(i,title=f"final_{i}")
            
    def time_speed_for_all(self,reps=30):
        import torch.utils.benchmark as benchmark
        times:list[list[float]]=[]
        times_back:list[list[float]]=[]
        times_nn:list[list[float]]=[]
        times_back_nn:list[list[float]]=[]
        for idx,data in enumerate(self.data.trainImages):
            scene_id,cam_type,camera,*unused =data
            t0=benchmark.Timer(
                setup='torch.cuda.synchronize()',
                stmt='rez=self.draw_subplots(*data);torch.cuda.synchronize()',
                globals={'self':self,'data':[scene_id,int(cam_type),camera,self.nn.required_subplots()]})
            metric0=t0.timeit(number=reps)
            times.append(metric0.mean)
            
            t1=benchmark.Timer(
                setup='rez=sum([x.mean() for x in self.draw_subplots(*data)]);torch.cuda.synchronize()',
                stmt='rez.backward(retain_graph=True);torch.cuda.synchronize()',
                globals={'self':self,'data':[scene_id,int(cam_type),camera,self.nn.required_subplots()]})
            metric1=t1.timeit(number=reps)
            times_back.append(metric1.mean)
            
            
            t2=benchmark.Timer(
                setup='torch.cuda.synchronize()',
                stmt='rez=[self.nn.forward(self.draw_subplots(*data))];torch.cuda.synchronize()',
                globals={'self':self,'data':[scene_id,int(cam_type),camera,self.nn.required_subplots()]})
            metric2=t2.timeit(number=reps//5 + 1)
            times_nn.append(metric2.mean)
            
            
            t3=benchmark.Timer(
                setup='rez=self.nn.forward(self.draw_subplots(*data));bck=torch.rand_like(rez);torch.cuda.synchronize()',
                stmt='rez.backward(bck,retain_graph=True);torch.cuda.synchronize()',
                globals={'self':self,'data':[scene_id,int(cam_type),camera,self.nn.required_subplots()]})
            metric3=t3.timeit(number=reps//5 + 1)
            times_back_nn.append(metric3.mean)
            print(f"Image {idx+1} benchmark complete")
        return times,times_back,times_nn,times_back_nn

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
    def __init__(self,parent,close_condition) -> None:
        super().__init__()
        self.should_stop_training=False
        self.parent:trainer=parent
        self.close_condition=close_condition
    def run(self):
        self.parent.nn.train()
        self.parent.data.train()
        kernelItf.initialize()
        last_save=last_report=s=time.time()
        unprinted_validation=False
        while(not self.should_stop_training):
            if(self.close_condition()):
                break
            if(args.samples_every>0 and (TOTAL_BATCHES_THIS_RUN%args.samples_every==0)):
                self.parent.save_one_()
            loss_this_batch=self.parent.train_one_batch()
            if(args.validation_interval>0 and TOTAL_BATCHES_THIS_RUN % args.validation_interval == 0):
                validation_loss_this_batch=self.parent.validate_one_batch()
                unprinted_validation=True
                #print(f"Validation loss:{validation_loss_this_batch}")
            e=time.time()
            if(e-last_report>args.report_freq or self.should_stop_training and self.parent.report_batches>0):
                l={loss:float(self.parent.report_losses[loss]/self.parent.report_batches) for loss in self.parent.report_losses}
                print(f"Report: average loss is {l} in {self.parent.report_batches} batches")
                if(unprinted_validation):
                    print(f"Last validation:{validation_loss_this_batch}")
                    unprinted_validation=False
                dt=e-s
                print(f"Total batches:{metaData.batches}(this run {TOTAL_BATCHES_THIS_RUN});Runtime={int(dt)//3600:02d}:{(int(dt)//60)%60:02d}:{int(dt)%60:02d}.{str(math.fmod(dt,1))[2:]}")
                
                l={}
                last_report=e
                self.parent.report_losses={}
                self.parent.report_batches=0
            if(e-last_save>args.autosave_s):
                tmp=time.time()
                print("autosaving...")
                last_save=time.time()
                print(f"autosaved ({last_save-tmp}s)")
                self.parent.save()
            if(args.stagnation_batches!=-1):
                append_loss_history(loss_this_batch)
        kernelItf.cleanup()
