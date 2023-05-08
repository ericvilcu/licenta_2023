import torch
import os
import customSer
import shutil
import args
def get_expansion_func(type:str):
    if(type=="norm"):
        return torch.randn
    if(type=="zero" or type=="zeros"):
        return torch.zeros
    if(type=="one" or type=="ones"):
        return lambda *dims,**kw:torch.ones(*dims,**kw)
    if(type=="-1"):
        return lambda *dims,**kw:torch.ones(*dims,**kw)*-1
    raise Exception(f"Unknown expansion func {type}")
class learnableData(torch.nn.Module):
    def __init__(self,fn:str=None,points:torch.Tensor=None,env:torch.Tensor=None,force_ndim=None) -> None:
        super().__init__()
        if(fn!=None):
            self.environment_type=-1
            if(os.path.exists(fn+"/points.txt")):
                points = customSer.from_text(fn+"/points.txt")
                environment = customSer.from_text_list(fn+"/environment.txt")
            elif(os.path.exists(fn+"/points.bin")):
                points = customSer.from_bin(fn+"/points.bin")
                environment = customSer.from_bin_list(fn+"/environment.bin")
            elif(os.path.exists(fn+"/points")):
                points = torch.load(fn+"/points")
                environment = torch.load(fn+"/environment")
            else: raise Exception("dataset location was invalid")
            #points[::,3:]+=torch.rand_like(points[::,3:])*1e-3
            self.environment_type=int(environment[0]-1)
            environment=environment[1:]
        else:
            points = points
            environment = env
        if(force_ndim!=None):
            ndim=len(points[0])-3
            if(ndim>force_ndim):
                #trunc
                trunc=ndim-force_ndim
                points=points[::,:-trunc:].contiguous()
                if(self.environment_type==0):
                    environment=points[::,::,::,:-trunc:].contiguous()
                elif(self.environment_type==1):
                    #TODO
                    environment=points[::,::,::,:-trunc:].contiguous()
                elif(self.environment_type==2):
                    environment=points[::,::,::,:-trunc:].contiguous()
            elif(force_ndim>ndim):
                ee=get_expansion_func(args.expand_environment)
                ep=get_expansion_func(args.expand_points)
                extra=force_ndim-ndim
                ps=list(points.shape)
                ps[-1]=extra
                append_points=ep(*ps,device='cuda')
                points: torch.Tensor=torch.cat((points,append_points),-1)
                append_points=None
                points=points.contiguous()
                
                #TODO: expand environment
                if(self.environment_type==0):
                    append_environment=ee((extra,),device='cuda')
                    environment[0]=torch.cat([environment[0][:-1],append_environment,environment[0][-1:]])
                elif(self.environment_type==1):
                    gen_append=lambda x:torch.cat([x[:-1],ee((extra,),device='cuda'),x[-1:]])
                    # 1. ground color (close)  
                    environment[0]=gen_append(environment[0])
                    # 2. ground color (horizon)
                    environment[1]=gen_append(environment[1])
                    # constexpr int GROUND_BLEND_IDX=GROUND2_IDX+NDIM+1;
                    # 3. ground horizon blending
                    # unmodified
                    # 4. sky color (close)
                    environment[3]=gen_append(environment[3])
                    # 5. sky color (horizon)
                    environment[4]=gen_append(environment[4])
                    # 6. sky horizon blending
                    # unmodified
                    # 7. sun position
                    # unmodified
                    # 8. sun color
                    environment[7]=gen_append(environment[7])
                    # 9. sun radius
                    # unmodified
                    # 10. sun sky contribution
                    environment[9]=gen_append(environment[9])
                    # 11. sun sky radius          (1)
                    # unmodified
                elif(self.environment_type==2):
                    environment=environment[0]
                    es=list(environment.shape)
                    es[-1]=extra
                    append_environment=torch.randn(*es,device='cuda')
                    environment: torch.Tensor=torch.cat((environment[:,:,:,:-1],append_environment,environment[:,:,:,-1:]),-1)#NOTE: depth is always the last one in line
                    append_environment=None
                    environment=[environment.contiguous()]
        import reorder
        self.environment:list[torch.Tensor]=torch.nn.ParameterList(environment)
        self.points:torch.Tensor=torch.nn.Parameter(reorder.reorder(points))
        environment=points=None
        
        self.points.requires_grad = args.refine_points
        self.environment.requires_grad = args.refine_points
        self.to('cuda')
    
    def get_environment(self):
        if(self.environment_type in (0,2)):
            return self.environment[0]
        else:
            return torch.cat([param.reshape(param.numel()).cuda() for param in self.environment])
        
    def save(self,path):
        points_path=os.path.join(path,"points.bin")
        environment_path=os.path.join(path,"environment.bin")
        customSer.to_bin(self.points,points_path)
        customSer.to_bin_list([torch.tensor([self.environment_type+1]),*self.environment],environment_path)


class Scene(torch.nn.Module):
    def __init__(self,fn:str=None,points:torch.Tensor=None,env:torch.Tensor=None,force_ndim=None) -> None:
        super().__init__()
        if(fn!=None):
            self.data = learnableData(fn=fn,force_ndim=force_ndim)
            self.path=fn
        else:
            self.data = learnableData(points=points,env=env,force_ndim=force_ndim)
            self.path=None
    def save(self,path,keepImages=True):
        self.data.save(path)
        if(self.path!=None and keepImages and self.path!=path):
            images_path=os.path.join(path,"train_images")
            if(not os.path.isdir(images_path)):
                if(os.path.exists(images_path)):
                    raise Exception(f"Path {images_path} is invalid")
                os.mkdir(images_path)
            old_images_dir=os.path.join(self.path,"train_images")
            for i in os.listdir(old_images_dir):
                shutil.copy(os.path.join(old_images_dir,i),images_path)
    def get_environment(self):
        return self.data.get_environment()



class DataSet(torch.nn.Module):
    def __init__(self,modules:list[Scene]=None,scenes:list[str]=None,force_ndim=None) -> None:
        super().__init__()
        if(modules!=None):
            self.scenes = torch.nn.ModuleList(modules)
        else:
            self.scenes = torch.nn.ModuleList(map(lambda path:Scene(fn=path,force_ndim=force_ndim),scenes))
        #for i,scene in enumerate(self.scenes):self.register_module(f"scene{i}",scene)
        self.trainImages=TrainImages(self)
    def save_to(self,path:str):
        for idx,scene in enumerate(self.scenes):
            scene_dir=os.path.join(path,str(idx))
            if(not os.path.isdir(scene_dir)):
                if(os.path.exists(scene_dir)):
                    raise Exception(f"Path {scene_dir} is invalid")
                os.mkdir(scene_dir)
            scene.save(scene_dir)
    @staticmethod
    def load(path:str):
        scenes=[]
        i=0
        while(True):
            if(os.path.isdir(os.path.join(path,str(i)))): scenes.append(os.path.join(path,str(i)))
            else: break
            i+=1
        return DataSet(scenes=scenes)

from torch.utils.data import Dataset
import struct
import io
def read_bytes(s:str,f:io.FileIO):
    return struct.unpack(s,f.read(struct.calcsize(s)))

class camera_updater():
    def __init__(self,cam_type,old_camera,path:list[str]):
        self.type=cam_type
        self.old=old_camera
        self.pth=path
    def __call__(self, new_camera):
        if(self.old.numel()!=new_camera.numel()):
            raise Exception('camera length not equal to previous camera')
        if(self.pth[0].endswith('.bin')):
            with open(self.pth[0],'r+b') as f:
                f.seek(0)
                f.write(struct.pack("i",self.type))
                f.write(struct.pack("IIII",int(new_camera[0]),int(new_camera[1]),int(new_camera[2]),int(new_camera[3])))
                f.write(struct.pack("f"*len(new_camera[4:]),*map(float,new_camera[4:])))


class TrainImages(Dataset):
    def __init__(self, ds:DataSet):
        super().__init__()
        self.paths:list[tuple[str,int]]=[]
        for scene_id,scene in enumerate(ds.scenes):
            pth=scene.path
            for suffix in ['.bin','.txt','']:
                if(pth!=None):
                    suffix 
                    i=0
                    file=os.path.join(pth,"train_images",f"{i:06d}"+suffix)
                    while(os.path.exists(file)):
                        self.paths.append((file,scene_id))
                        i+=1
                        file=os.path.join(pth,"train_images",f"{i:06d}"+suffix)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx:int) -> tuple[int, int, list[float], torch.Tensor]:
        #TODO: maybe implement formats for like a png and a camera.txt
        fn,scene_id=self.paths[idx]
        if(fn.endswith('.txt')):
            raise Exception("did not implement reading text images")
        elif(fn.endswith('.bin')):
            with open(fn,'rb') as f:
                cam_type:int
                cam_type,w0,h0,w,h=read_bytes('IIIII',f)
                lum,=read_bytes('f',f)
                rot=read_bytes('f'*9,f)
                rot=(rot[0],rot[1],rot[2],
                            rot[3],rot[4],rot[5],
                            rot[6],rot[7],rot[8])
                trans=read_bytes('f'*3,f)
                trans=(trans[0],trans[1],trans[2])
                
                
                if(cam_type == 0):
                    #NOTE: ppx,ppy,fx,fy
                    ppx,ppy,fx,fy=read_bytes('f'*4,f)
                    extra=[ppx,ppy,fx,fy]
                    
                else:raise Exception("Unknown camera type")
                cam=torch.tensor(list(map(float,[w0,h0,w,h,lum,*rot,*trans,*extra])))
                image_data=bytearray(f.read())
                target=(torch.frombuffer(image_data,dtype=torch.uint8).cuda().reshape((h,w,4))).to(dtype=torch.float32)/255
                #It should be a w x h x 4 data thing
                return scene_id,cam_type,cam,target,{'cam_type':cam_type,'old_camera':cam,'path':fn}
        else:
            raise Exception("did not implement reading torch images")


class SubDataSet(Dataset):
    def __init__(self,img:TrainImages,take=1,skip=0,invert=False) -> None:
        super().__init__()
        self.all_data=img
        self.take=take
        self.skip=skip
        self.invert=invert
    def __len__(self):
        #NOT instant? todo? precalculation?
        l1=len(self.all_data)
        wrp=l1//(self.take+self.skip)
        rem=l1%(self.take+self.skip)
        tk=self.skip if self.invert else self.take
        rem=rem-min(rem,self.take) if self.invert else min(rem,self.take)
        return wrp*tk+rem
    def __getitem__(self, idx:int):
        siz=self.take+self.skip
        off=(self.take if self.invert else 0)
        p=(self.skip if self.invert else self.take)
        idx_w=idx//p
        idx_r=idx%p
        return self.all_data.__getitem__(off + siz * idx_w + idx_r)
    
# def testSubDataSet():
#     from torch.utils.data import DataLoader
#     class tmpMock(Dataset):
#         def __init__(self,sz=20) -> None:
#             super().__init__()
#             self.sz=20
#         def __getitem__(self, index):
#             return index
#         def __len__(self):
#             return self.sz
    
#     ds=tmpMock(20)
#     all=DataLoader(SubDataSet(tmpMock(20)),batch_size=1, shuffle=False)
#     trn=DataLoader(SubDataSet(ds,10,2,False), batch_size=1, shuffle=True)
#     val=DataLoader(SubDataSet(ds,10,2,True ), batch_size=1, shuffle=False)
#     assert(len(all)>len(val))
#     assert(len(all)>len(trn))
#     assert(len(trn)>len(val))
#     assert(len(ds)==len(all))
#     assert(len(trn)+len(val)==len(all))
#     trn_s=set(map(int,trn))
#     val_s=set(map(int,val))
#     all_s=set(map(int,all))
#     assert(len(trn_s)==len(trn))
#     assert(len(val_s)==len(val))
#     assert(len(all_s)==len(all))
    
#     assert(len(all_s)==len(trn_s.union(val_s)))
#     assert(len(trn_s.intersection(val_s))==0)
    
    
    
#     pass

# testSubDataSet()
