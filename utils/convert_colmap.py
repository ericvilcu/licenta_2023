import os
from scipy.spatial.transform import Rotation
import numpy as np
import struct
from itertools import chain

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
def patchImages(cameras_txt,images_txt,custom_bin_image_folder,overwrite=False,img_location=None,end=".bin"):
    done=0
    if(overwrite):
        if(img_location==None or len(img_location)<2):
            raise Exception("Please provide an image path if overwrite is true such that the specified files can be found.")
    img_data = {int(i.strip().split(' ')[0]):i.strip().split(' ') for i in all_lines(images_txt) if (i.strip().endswith('.png') or i.strip().endswith(".jpg")) and not i.startswith("#")}
    cam_data ={int(i.strip().split(' ')[0]):i.strip().split(' ') for i in all_lines(cameras_txt) if not i.startswith('#')}
    PN=2# prints this many transforms for debugging purposes.
    for idx in img_data:
        img = img_data[idx]
        img_id=img[0]
        #quaternions;-;
        QW, QX, QY, QZ, TX, TY, TZ= map(float,img[1:-2])
        r:Rotation = Rotation.from_quat([QW,QZ,QY,QX,])#This is the one that makes the image align. I do not know why. Perhaps someone with more knowledge of quaternions would help.
        rotation_mat = np.array(r.as_matrix())
        #I honestly have no clue what is going on here. It is just the formula that seems to lead to the correct results and the plot aligning with the camera.
        #I can only blame my poor understanding of quaternions and the systems around them.
        translation = np.matmul(-rotation_mat.transpose(),np.array([TX,TY,-TZ]))
        translation=-np.array([translation[0],translation[1],translation[2]])        
        rotation_mat = -rotation_mat.transpose()
        translation = np.matmul(rotation_mat,translation)
        
        #translation = -np.array([TX,TY,TZ])
        if PN>0:
            print(img)
            print(*[rotation_mat,translation,[TX,TY,TZ],[QW,QZ,QY,QX]],sep='\n');PN-=1
        #rotation_mat = rotation_mat.transpose()
        cam_id= int(img[-2])
        img_name = img[-1]
        cam = cam_data[cam_id]
        _cam_id = cam[0]
        type = cam[1]
        assert(type=="PINHOLE")
        w,h,fx,fy,ppx,ppy=map(float,cam[2:])
        path_to_patch=os.path.join(custom_bin_image_folder,no_extension(img_name)+end)
        if(not overwrite):
            if os.path.exists(path_to_patch):
                with open(path_to_patch,"r+b") as f:
                    f.write(struct.pack("i",0))
                    f.write(struct.pack("f"*12,*chain(*rotation_mat),*translation))
                    f.write(struct.pack(4*"f",ppy,ppx,fy,fx))
                    n=h
                    m=w
                    f.write(struct.pack("II",int(n),int(m)))
        else:
            with open(path_to_patch,"wb") as f:
                f.write(struct.pack("i",0))
                f.write(struct.pack("f"*12,*chain(*rotation_mat),*translation))
                f.write(struct.pack(4*"f",ppy,ppx,fy,fx))
                n=h
                m=w
                f.write(struct.pack("II",int(n),int(m)))
                from PIL import Image
                img_raw = Image.open(os.path.join(img_location,img_name))
                #"We have .bmp at home"
                for j in range(int(n)):
                    for i in range(int(m)):
                        pix=img_raw.getpixel((i,j))
                        R,G,B=pix
                        A=255
                        f.write(struct.pack("B"*4,R,G,B,A))
                if(done&0xf==0xf):
                    print(done+1,"images done")
        done+=1
                    
def patch_points(points_txt,custom_bin_path, extra_color_channels=0):
    with open(custom_bin_path,"wb") as f:
        f.write(struct.pack("qq",1,(3+3+extra_color_channels)))
        for ln in all_lines_iter(points_txt):
            ln=ln.strip()
            if(len(ln)>0 and not ln.startswith('#')):
                POINT3D_ID, X, Y, Z, \
                    R, G, B,err, *track \
                        = map(float,ln.split(' '))
                f.write(struct.pack("fff", X, Y, Z))
                f.write(struct.pack("fff"+("f"*extra_color_channels), *map(lambda x:x/255,[R, G, B] +([0]*extra_color_channels))))

def make_dummy_environment(custom_bin_path, extra_color_channels:int=0,resolution:int=1024):
    #We may also want to make a dummy environment    
    with open(custom_bin_path,"wb") as f:
        f.write(struct.pack("qqqqq",4,6,resolution,resolution,4+extra_color_channels))
        for face in range(6):
            for x in range(resolution):
                for y in range(resolution):
                    f.write(struct.pack("fff"+("f"*extra_color_channels) + "f", *map(lambda x:x/255,[0,0,0] +([0]*extra_color_channels)+[0])))
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
    
    DO_PNTS=bool(int(argv[4]))
    DO_IMGS=bool(int(argv[5]))
    CONVERT_WHOLE_IMAGE=bool(int(argv[6]))
    
    OUT_DATASET_LOCATION= argv[7]
    IMGS=f"{OUT_DATASET_LOCATION}/train_images"
    PNTS=f"{OUT_DATASET_LOCATION}/points.bin"
    ENVD=f"{OUT_DATASET_LOCATION}/environment.bin"
    
    start=time.time()
    if(CONVERT_WHOLE_IMAGE):
        if(not os.path.exists(OUT_DATASET_LOCATION)):
            os.mkdir(OUT_DATASET_LOCATION)
        if(not os.path.exists(IMGS)):
            os.mkdir(IMGS)
    if(DO_PNTS):
        patch_points(PNTS_TXT,PNTS)
        print("Points transferred.")
        make_dummy_environment(ENVD)
        print("Dummy environment created.")
    if(DO_IMGS):
        patchImages(CAM_TXT,IMG_TXT,IMGS,overwrite=CONVERT_WHOLE_IMAGE,img_location=IMG_LOCATION)
    end=time.time()
    #Last time it took about 1s per frame with all options, which is slow, but that's python and this was really easy to write.
    print(f"Done({'points' if DO_PNTS else ''}{'+' if DO_PNTS and DO_IMGS else ''}{f'images (whole={CONVERT_WHOLE_IMAGE})' if DO_IMGS else ''}) took {end-start}s")