import os
import sys
import shutil
"""All this script does is copy files to a directory, giving them new names. I use this to mae the kitti dataset interface better with what i am doing
"""
pj=os.path.join
if(len(sys.argv)==1):
    KITTI_DIR=input("kitti dir=")
    OUT_DIR=input("out dir=")
else:
    KITTI_DIR=sys.argv[1]
    OUT_DIR=sys.argv[2]

L_DIR=pj(KITTI_DIR,"image_2")
R_DIR=pj(KITTI_DIR,"image_3")

for idx,(iml,imr) in enumerate(zip(os.listdir(L_DIR),os.listdir(R_DIR))):
    id_l=idx*2
    id_r=idx*2+1
    shutil.copyfile(pj(L_DIR,iml),pj(OUT_DIR,f"{id_l:06d}.png"))
    shutil.copyfile(pj(R_DIR,imr),pj(OUT_DIR,f"{id_r:06d}.png"))