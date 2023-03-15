import torch
import os
import customSer
class learnableData(torch.nn.Module):
    def __init__(self,fn:str=None,points:torch.Tensor=None,env:torch.Tensor=None) -> None:
        super().__init__()
        if(fn!=None):
            if(os.path.exists(fn+"/points.txt")):
                self.points = customSer.from_text(fn+"/points.txt")
                self.environment = customSer.from_text(fn+"/environment.txt")
            elif(os.path.exists(fn+"/points.bin")):
                self.points = customSer.from_bin(fn+"/points.bin")
                self.environment = customSer.from_bin(fn+"/environment.bin")
            elif(os.path.exists(fn+"/points")):
                self.points = torch.load(fn+"/points")
                self.environment = torch.load(fn+"/environment")
            else: raise Exception("dataset location was invalid")
        else:
            self.points = points
            self.environment = env

class Scene(torch.nn.Module):
    def __init__(self,fn:str=None,points:torch.Tensor=None,env:torch.Tensor=None) -> None:
        super().__init__()
        if(fn!=None):
            self.data = learnableData(fn=fn)
            self.path=fn
        else:
            self.data = learnableData(points=points,env=env)
            self.points = points
            self.env = env
            self.path=None

class DataSet(torch.nn.Module):
    def __init__(self,modules:list=None,scenes:list[str]=None) -> None:
        super().__init__()
        if(modules!=None):
            self.scenes = modules
        else:
            self.scenes = list(map(lambda path:Scene(fn=path),scenes))
        for i,scene in enumerate(self.scenes):self.register_module(f"scene{i}",scene)
        self.trainImages=TrainImages(self)

from torch.utils.data import Dataset
import struct
import io
def read_bytes(s:str,f:io.FileIO):
    return struct.unpack(s,f.read(struct.calcsize(s)))
#This class alone may urge me to use asyncIO
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

    def __getitem__(self, idx) -> tuple[int, int, list[float], torch.Tensor]:
        #TODO: maybe implement formats for like a png and a camera.txt
        fn,scene_id=self.paths[idx]
        if(fn.endswith('.txt')):
            raise Exception("did not implement reading text images")
        elif(fn.endswith('.bin')):
            with open(fn,'rb') as f:
                cam_type:int
                cam_type,w0,h0,w,h=read_bytes('IIIII',f)
                rot=read_bytes('f'*9,f)
                trans=read_bytes('f'*3,f)
                if(cam_type == 0):
                    #NOTE: fx,fy,ppx,ppy
                    extra=read_bytes('f'*4,f)
                    pass
                else:raise Exception("Unknown camera type")
                cam=list(map(float,[w0,h0,w,h,*rot,*trans,*extra]))
                image_data=bytearray(f.read())
                target=(torch.frombuffer(image_data,dtype=torch.uint8).cuda().reshape((h,w,4))).to(dtype=torch.float32)/255
                #It should be a w x h x 4 data thing
                return scene_id,cam_type,cam,target
        else:
            raise Exception("did not implement reading torch images")

