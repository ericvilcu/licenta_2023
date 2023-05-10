import os
import sys
import shutil
"""All this script does is copy files to a directory, giving them new names. I use this to make the kitti dataset interface better with what i am doing
"""
pj=os.path.join
if(len(sys.argv)==1):
    KITTI_DIR=input("kitti dir=")
    OUT_DIR=input("out dir=")
    LIMIT=input("LIMIT=")
else:
    KITTI_DIR=sys.argv[1]
    OUT_DIR=sys.argv[2]
    LIMIT=sys.argv[3] if len(sys.argv)>3 else 'no'
try:
    LIMIT=int(LIMIT)
except Exception:
    LIMIT=-1

L_DIR=pj(KITTI_DIR,"image_2")
R_DIR=pj(KITTI_DIR,"image_3")

for idx,(iml,imr,*unused) in enumerate(zip(os.listdir(L_DIR),os.listdir(R_DIR),*[range(LIMIT) for __ in range(1) if LIMIT>-1])):
    id_l=idx*2
    id_r=idx*2+1
    shutil.copyfile(pj(L_DIR,iml),pj(OUT_DIR,f"{id_l:06d}.png"))
    shutil.copyfile(pj(R_DIR,imr),pj(OUT_DIR,f"{id_r:06d}.png"))