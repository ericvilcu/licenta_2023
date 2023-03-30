#And here I thought I had decent abstractions for once.
#Maybe I'll manage change my camera models somehow so there's only one place where you need to change them.


def downsize_pinhole(cam_data:list[float])->list[float]:
    w0,h0,w,h,lum,r1,r2,r3,r4,r5,r6,r7,r8,r9,t1,t2,t3,fx,fy,ppx,ppy= \
        cam_data
    fx=fx/2;fy=fy/2
    ppx=ppx/2;ppy=ppy/2
    w0=int(w0/2);w=int(w/2)
    h0=int(h0/2);h=int(h/2)
    return torch.tensor([w0,h0,w,h,lum,r1,r2,r3,r4,r5,r6,r7,r8,r9,t1,t2,t3,fx,fy,ppx,ppy])

downsize_functions={
    0:downsize_pinhole
}

def downsize_camera(cam_type:int,cam_data:list[float]):
    return downsize_functions[int(cam_type)](cam_data)
def pad_camera(cam_data:list[float],pad):
    copy=[float(i) for i in cam_data] if type(cam_data)==list else cam_data.clone()
    copy[0]-=pad
    copy[1]-=pad
    copy[2]+=2*pad
    copy[3]+=2*pad
    return copy
    


import torch
import math
def best_split_camera(camera,MAX_PIXELS_PER_PLOT,expected_pad=0):
    """Splits camera into smaller cameras such that any camera's size, with padding expected_pad, does not exceed MAX_PIXELS_PER_PLOT

    Returns a matrix of cameras, and the maximum width/height of each (including padding)
    """
    #By convention, camera data starts with W0,H0,W and H, this will be used to simplify calculations.
    wm,hm=1,1
    W0,H0,W,H=map(int,camera[0:4])
    wm=hm=int(math.sqrt(W*H/MAX_PIXELS_PER_PLOT))
    ew,eh=math.ceil(W/wm),math.ceil(H/hm)
    while((ew+2*expected_pad)*(eh+2*expected_pad)>MAX_PIXELS_PER_PLOT):
        if(wm>hm):hm=hm+1
        else:wm=wm+1
        ew,eh=math.ceil(W/wm),math.ceil(H/hm)
    
    cams:list[list[list[float]]]=[]
    for i in range(wm): 
        row=[]
        for j in range(hm): 
            cpy=[float(i) for i in camera] if type(camera)==list else camera.clone()
            w0,h0,w1,h1=(W0+(i)*H//wm),(H0+(j)*H//hm),(W0+(i+1)*W//wm),(H0+(j+1)*H//hm)
            w,h=w1-w0,h1-h0
            cpy[0]=w0
            cpy[1]=h0
            cpy[2]=w
            cpy[3]=h
            row.append(cpy)
        cams.append(row)
    return cams,ew+2*expected_pad,eh+2*expected_pad

def tensor_subsection(tsr:torch.Tensor,cam:list[float]):
    w0,h0,w,h,*unused=cam
    
    return tsr[int(h0):int(h0)+int(h),int(w0):int(w0)+int(w)]
def unpad_tensor(tsr:torch.Tensor,padding:int):
    return tsr[padding:-padding,padding:-padding]
def put_back_together(tensors):
    tsr=torch.zeros((0,))
    for row in tensors:
        tsr_row=torch.zeros((0,))
        for col in row:
            torch.cat(tsr,tsr_row,0)
        torch.cat(tsr,tsr_row,0)
    return tsr