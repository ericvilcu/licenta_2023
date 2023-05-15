import os
from scipy.spatial.transform import Rotation
import numpy as np
import struct
from itertools import chain

def clamp(x,mn,mx):
    return min(max(x,mn),mx)

def no_extension(fn:str):
    for i in range(len(fn)-1,-1,-1):
        if(fn[i]=='.'):
            return fn[:i]
    return fn
def all_lines_iter(file:str):
    with open(file) as f:
        l=f.readline()
        while(len(l)>0):
            yield l
            l=f.readline()
    return
def all_lines(file:str):
    with open(file) as f:
        return f.readlines()
def patchImages(cameras_txt,images_txt,custom_bin_image_folder,overwrite=False,img_location=None,end=".bin",autoReorder=False):
    """
        returns an average color to use in defining the environment.
    """
    if(autoReorder):
        IDX=0
    r=g=b=ncol=0
    done=0
    if(overwrite):
        if(img_location==None or len(img_location)<2):
            raise Exception("Please provide an image path if overwrite is true such that the specified files can be found.")
    img_data = {int(i.strip().split(' ')[0]):i.strip().split(' ') for i in all_lines(images_txt) if (i.strip().lower().endswith('.png') or i.strip().lower().endswith(".jpg") or i.strip().lower().endswith(".jpeg")) and not i.startswith("#")}
    cam_data ={int(i.strip().split(' ')[0]):i.strip().split(' ') for i in all_lines(cameras_txt) if not i.startswith('#')}
    PN=2# prints this many transforms for debugging purposes.
    for idx in img_data:
        img = img_data[idx]
        img_id=img[0]
        #quaternions;-;
        QW, QX, QY, QZ, TX, TY, TZ= map(float,img[1:-2])
        #In colmap, quaternions are based on the hamilton convention https://fzheng.me/2017/11/12/quaternion_conventions_en/
        rot:Rotation = Rotation.from_quat([QW,QZ,QY,QX,])#This is the one that makes the image align. I do not know why. Perhaps someone with more knowledge of quaternions would help.
        rotation_mat = -np.array(rot.as_matrix()).transpose()
        #I honestly have no clue what is going on here. It is just the formula that seems to lead to the correct results and the plot aligning with the camera.
        #I can only blame my poor understanding of quaternions and the systems around them.
        #I think I know what I'm doing here now. I'm inverting the matrix in a really convoluted way because it is a matrix meant to be multiplied to the left, and I'm multiplying it to the right.
        #TBH now that I know that I want to replace it with a custom transform class with a 3x3 rotation and 1x3 translation
        #still not sure what those minuses are about 
        #translation = np.matmul(-rotation_mat.transpose(),np.array([TX,TY,-TZ]))
        #translation=-np.array([translation[0],translation[1],translation[2]])        
        #rotation_mat = -rotation_mat.transpose()
        #translation = np.matmul(rotation_mat,translation)
        #according to https://colmap.github.io/format.html#images-txt, translation is on the camera coordinate system.
        translation = np.array([TX,TY,-TZ])
        translation = -np.matmul(rotation_mat,translation)
        
        if PN>0:
            print(img)
            print(*[rotation_mat,translation,[TX,TY,TZ],[QW,QZ,QY,QX]],sep='\n');PN-=1
        #rotation_mat = rotation_mat.transpose()
        cam_id= int(img[-2])
        img_name = img[-1]
        cam = cam_data[cam_id]
        _cam_id = cam[0]
        type = cam[1]
        assert(type in ["PINHOLE","SIMPLE_PINHOLE","RADIAL","SIMPLE_RADIAL"])
        if(type == "PINHOLE"):
            t=0
            w,h,fx,fy,ppx,ppy=map(float,cam[2:])
            w0=h0=0
        elif(type == "SIMPLE_PINHOLE"):
            t=0
            w,h,fx,ppx,ppy=map(float,cam[2:])
            w0=h0=0
            fy=fx
        elif(type == "RADIAL"):
            t=1
            w,h,fx,ppx,ppy,k1,k2=map(float,cam[2:])
            w0=h0=0
            fy=fx
        elif(type == "SIMPLE_RADIAL"):
            t=1
            w,h,fx,ppx,ppy,k1=map(float,cam[2:])
            w0=h0=0
            fy=fx;k2=0.
        else:raise f"{type} is an unknown camera type."
        if(autoReorder):
            path_to_patch=os.path.join(custom_bin_image_folder,str(IDX)+end)
            IDX+=1
        else: path_to_patch=os.path.join(custom_bin_image_folder,no_extension(img_name)+end)
        lum=1.0
        if(not overwrite):
            if os.path.exists(path_to_patch):
                with open(path_to_patch,"r+b") as f:
                    f.write(struct.pack("i",t))
                    f.write(struct.pack("IIII",int(w0),int(h0),int(w),int(h)))
                    f.write(struct.pack("f"*13,lum,*chain(*rotation_mat),*translation))
                    if(t==0):#Note: changing camera models with patch destroys stuff if its dimensions change, but since I'm not really using patch mode anymore, that may be ok.
                        f.write(struct.pack(4*"f",ppx,ppy,fx,fy))
                    elif(t==1):
                        f.write(struct.pack(6*"f",ppx,ppy,fx,fy,k1,k2))
        else:
            with open(path_to_patch,"wb") as f:
                f.write(struct.pack("i",t))
                f.write(struct.pack("IIII",int(w0),int(h0),int(w),int(h)))
                f.write(struct.pack("f"*13,lum,*chain(*rotation_mat),*translation))
                if(t==0):#Note: changing camera models with patch destroys stuff, but since I'm not really using patch mode anymore, that may be ok.
                    f.write(struct.pack(4*"f",ppx,ppy,fx,fy))
                elif(t==1):
                    f.write(struct.pack(6*"f",ppx,ppy,fx,fy,k1,k2))
                n=h
                m=w
                from PIL import Image
                img_raw = Image.open(os.path.join(img_location,img_name))
                #"We have .bmp at home"
                for j in range(int(n)):
                    for i in range(int(m)):
                        pix=img_raw.getpixel((i,j))
                        R,G,B=pix
                        A=255
                        r+=(R/255);g+=(G/255);b+=(B/255);ncol+=1
                        f.write(struct.pack("B"*4,R,G,B,A))
                if(done&0xf==0xf):
                    print(done+1,"images done")
        done+=1
    return (r/ncol,g/ncol,b/ncol) if ncol>0 else (0,0,0)
                    
def patch_points(points_txt,custom_bin_path, extra_color_channels=0):
    """
        returns an average color to use in defining the environment.
    """
    r=g=b=ncol=0
    with open(custom_bin_path,"wb") as f:
        f.write(struct.pack("qq",1,(3+3+extra_color_channels)))
        for ln in all_lines_iter(points_txt):
            ln=ln.strip()
            if(len(ln)>0 and not ln.startswith('#')):
                POINT3D_ID, X, Y, Z, \
                    R, G, B,err, *track \
                        = map(float,ln.split(' '))
                r+=(R/255);g+=(G/255);b+=(B/255);ncol+=1
                f.write(struct.pack("fff", X, Y, Z))
                f.write(struct.pack("fff"+("f"*extra_color_channels), *map(lambda x:x/255,[R, G, B] +([0]*extra_color_channels))))
    return r/ncol,g/ncol,b/ncol

def make_dummy_environment(custom_bin_path,type=3, extra_color_channels:int=0,resolution:int=512,clr=(0,0,0),depth:int=1e6,extra=None):
    #We may also want to make a dummy environment
    clr = list(map(lambda x:clamp(x,0,1),clr))
    print("Color is:",clr)
    if(extra == None):extra = [0]*extra_color_channels
    with open(custom_bin_path,"wb") as f:
        if(type==0):
            f.write(struct.pack("q",1))
            f.write(struct.pack("qqf",1,1,type))
        elif(type==1):
            f.write(struct.pack("q",2))
            f.write(struct.pack("qqf",1,1,type))
            
            f.write(struct.pack("qq",1,4+extra_color_channels))
            for i in range(3):
                f.write(struct.pack("f",clr[i]))
            for i in range(extra_color_channels):
                f.write(struct.pack("f",0.))
            f.write(struct.pack("f",depth))
        elif(type==3):
            f.write(struct.pack("q",2))
            f.write(struct.pack("qqf",1,1,type))
            
            f.write(struct.pack("qqqqq",4,6,resolution,resolution,4+extra_color_channels))
            for face in range(6):
                for x in range(resolution):
                    for y in range(resolution):
                        #clr = [face%2,(face//2) % 2,(face//4)%4]
                        #clr = [face/6,x/resolution,y/resolution]
                        f.write(struct.pack("fff"+("f"*extra_color_channels) + "f", *map(float,clr+([0]*extra_color_channels)+[depth])))
        elif(type==2):
            dnidm=4+extra_color_channels
            ylw=[1,1,.7]
            dclrs =[*clr]+[0.]*extra_color_channels+[1e6]
            dclrs_h =[d*0.7 for d in dclrs  [ :-1]]+[1e6]
            dclrs_g =[d*0.4 for d in dclrs  [ :-1]]+[1e6]
            dclrs_hg=[d*0.7 for d in dclrs_g[ :-1]]+[1e6]
            dclrs_s=ylw+ [0 for d in dclrs  [3:-1]]+[1e6]
            dclrs_sc=[d*0.7 for d in dclrs_s[ :-1]]+[1e6]
            f.write(struct.pack("q",13))#13 sub-tensors
            f.write(struct.pack("qqf",1,1,type))                  #TYPE
            
            f.write(struct.pack("qq"+"f"*dnidm,1,dnidm,*dclrs_g ))#GROUND
            f.write(struct.pack("qq"+"f"*dnidm,1,dnidm,*dclrs_hg))#GROUND2 (near horizon)
            f.write(struct.pack("qq"+"f",1,1,1))                  #GROUND BLEND
            
            f.write(struct.pack("qq"+"f"*dnidm,1,dnidm,*dclrs))   #SKY
            f.write(struct.pack("qq"+"f"*dnidm,1,dnidm,*dclrs_h)) #SKY2    (near horizon)
            f.write(struct.pack("qq"+"f",1,1,1))                  #SKY BLEND
            
            f.write(struct.pack("qq"+"f"*2,1,2,0,0))              #SUN POSITION
            f.write(struct.pack("qq"+"f"*dnidm,1,dnidm,*dclrs_s)) #SUN COLOR
            f.write(struct.pack("qq"+"f",1,1,.1))                 #SUN RADIUS
            f.write(struct.pack("qq"+"f"*dnidm,1,dnidm,*dclrs_sc))#SUN SKY CONTRIBUTION COLOR
            f.write(struct.pack("qq"+"f",1,1,.5))                 #SUN SKY CONTRIBUTION RADIUS
            f.write(struct.pack("qq"+"f",1,1,1))                  #SUN SKY CONTRIBUTION BLEND
            
            pass
"""
Not sure if this is worth automating, I may want to configure some things manually for best speed/results
"""
def applyColMapToImages(pth:str):
    #TODO?
    pass


if __name__ == "__main__":
    import time
    from sys import argv
    USE_TRANSPARENCY=True
    #   0            1            2              3               4              5              6               7
    #script.py "image_src" "sparse_folder" "dense_folder" "1/0 (points)" "1/0 (images)" "1/0(whole/patch)"" "output"
    IMG_LOCATION = argv[1]
    
    SPARSE_LOCATION=argv[2]
    IMG_TXT=f"{SPARSE_LOCATION}/images.txt"
    CAM_TXT=f"{SPARSE_LOCATION}/cameras.txt"
    PNTS_TXT=f"{argv[3]}/points3D.txt"
    
    DO_ENV=DO_PNTS=bool(int(argv[4]))
    DO_IMGS=bool(int(argv[5]))
    CONVERT_WHOLE_IMAGE=bool(int(argv[6]))
    
    OUT_DATASET_LOCATION= argv[7]
    IMGS=f"{OUT_DATASET_LOCATION}/train_images"
    PNTS=f"{OUT_DATASET_LOCATION}/points.bin"
    ENVD=f"{OUT_DATASET_LOCATION}/environment.bin"
    
    rgbp=rgbi=None
    
    start=time.time()
    if(CONVERT_WHOLE_IMAGE):
        if(not os.path.exists(OUT_DATASET_LOCATION)):
            os.mkdir(OUT_DATASET_LOCATION)
        if(not os.path.exists(IMGS)):
            os.mkdir(IMGS)
    if(DO_PNTS):
        rgbp=patch_points(PNTS_TXT,PNTS)
        print("Points transferred.")
    if(DO_IMGS):
        rgbi=patchImages(CAM_TXT,IMG_TXT,IMGS,overwrite=CONVERT_WHOLE_IMAGE,img_location=IMG_LOCATION)
        print("Images converted.")
    if(DO_ENV):
        est_frac = 1/2#what % of unreachable points exist, to choose a better sky color.
        if(rgbp == None):
            if(rgbi == None): clr=(0,0,0)
            else: clr=rgbi
        else:
            if(rgbi == None): clr=rgbp
            else: clr = list(map(lambda pi:(pi[1]-pi[0]*est_frac)/est_frac,zip(rgbp,rgbi)))
        print("colors:",rgbp,rgbi,clr)
        make_dummy_environment(ENVD,clr=clr)
        print("Dummy environment created.")
    end=time.time()
    #Last time it took about 1.5s per frame with all options, which is slow, but that's python and this was really easy to write.
    print(f"Done({'points' if DO_PNTS else ''}{'+' if DO_PNTS and DO_IMGS else ''}{f'images(whole={CONVERT_WHOLE_IMAGE})' if DO_IMGS else ''}) took {end-start}s")