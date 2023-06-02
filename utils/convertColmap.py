import time
from sys import argv,stderr
from convertColmapOLD import *
def crash(msg):
    print(msg,file=stderr)
    exit(-1)
if(len(argv)<=1):
    print("some paths will be required.")
    IMG_LOCATION=input("image folder location:")
    SPARSE_LOCATION=input("Location of sparse point cloud .txt (incl. cameras):")
    DENSE_LOCATION=input("Location of dense point cloud .txt:")
    CONVERT_POINTS=True#input("Convert points[(y)/n]?") in ['y','Y','','yes','Yes']
    CONVERT_ENVIRONMENT=True#input("Make dummy environment[(y)/n]?") in ['y','Y','','yes','Yes']
    CONVERT_IMAGES=True
    PATCH_IMAGES=False
    
    ENVIRONMENT_TYPE=input("Environment type[-1,(0),1,2]:")
    if(ENVIRONMENT_TYPE==''):ENVIRONMENT_TYPE='0'
    else:ENVIRONMENT_TYPE=int(ENVIRONMENT_TYPE)
    MASK_PATH=input("MASK PATH(leave blank for None):")
    OUTPUT_FOLDER=input("scene location?")
else:
    import argparse
    parser=argparse.ArgumentParser(prog="COLMAP converter")
    
    parser.add_argument("--img_location",required=True,help="Specify a folder full of jpgs or pngs that have been used in the reconstruction")
    parser.add_argument("--sparse_location",required=True,help="Specify the folder where you exported the sparse model as text")
    parser.add_argument("--dense_location",required=True,help="Specify the folder where you exported the dense model as text")
    parser.add_argument("--skip_points",required=False,default=False,action='store_true',help="add to skip point conversion (not recommended)")
    parser.add_argument("--skip_environment",required=False,default=False,action='store_true',help="add to skip dummy environment creation (not recommended)")
    parser.add_argument("--skip_images",required=False,default=False,action='store_true',help="add to skip image conversion (not recommended)")
    parser.add_argument("--patch_images",required=False,default=False,action='store_true',help="add to patch image cameras instead of converting them whole(not recommended)")
    parser.add_argument("--environment_type",required=False,default=0,help="Specify the environment type (default=0)",type=int)
    parser.add_argument("--mask",required=False,default=None,help="Specify a folder full of jpgs or pngs that have been used as masks in the reconstruction (optional)")
    parser.add_argument("--output",required=True,help="Specify where to put the converted scenes")
    
    raw_args=parser.parse_args()
    IMG_LOCATION=raw_args.img_location
    SPARSE_LOCATION=raw_args.sparse_location
    DENSE_LOCATION=raw_args.dense_location
    CONVERT_POINTS=not raw_args.skip_points
    CONVERT_ENVIRONMENT=not raw_args.skip_environment
    CONVERT_IMAGES=not raw_args.skip_images and not raw_args.patch_images
    PATCH_IMAGES=raw_args.patch_images
    
    ENVIRONMENT_TYPE=int(raw_args.environment_type)
    MASK_PATH=raw_args.mask
    OUTPUT_FOLDER=raw_args.output

if(MASK_PATH==''):MASK_PATH=None
assert ENVIRONMENT_TYPE in (-1,0,1,2), "invalid environment"



IMG_TXT=f"{SPARSE_LOCATION}/images.txt"
CAM_TXT=f"{SPARSE_LOCATION}/cameras.txt"
PNTS_TXT=f"{DENSE_LOCATION}/points3D.txt"




IMGS=f"{OUTPUT_FOLDER}/train_images"
PNTS=f"{OUTPUT_FOLDER}/points.bin"
ENVD=f"{OUTPUT_FOLDER}/environment.bin"

rgbp=rgbi=None
    
if(not os.path.exists(OUTPUT_FOLDER)):
    os.mkdir(OUTPUT_FOLDER)
if(not os.path.exists(IMGS)):
    os.mkdir(IMGS)

if(CONVERT_POINTS):
    if(not os.path.exists(PNTS_TXT)):
        crash(f"{PNTS_TXT} does not exist")
    rgbp=patch_points(PNTS_TXT,PNTS)
    print("Points transferred.")
if(CONVERT_IMAGES or PATCH_IMAGES):
    if(CONVERT_IMAGES and PATCH_IMAGES):
        crash("cannot both convert and patch images.")
    if(not os.path.exists(CAM_TXT)):
        crash(f"{CAM_TXT} does not exist.")
    if(not os.path.exists(IMG_TXT)):
        crash(f"{IMG_TXT} does not exist.")
    rgbi=patchImages(CAM_TXT,IMG_TXT,IMGS,overwrite=not PATCH_IMAGES,img_location=IMG_LOCATION,mask_location=MASK_PATH,autoReorder=True)
    print("Images converted.")
if(CONVERT_ENVIRONMENT):
    est_frac = 1/3#what % of unreachable points exist, to choose a better sky color.
    
    if(rgbp == None):
        if(rgbi == None): clr=(.5,.5,.5)
        else: clr=rgbi
    else:
        if(rgbi == None): clr=rgbp
        else: clr = list(map(lambda pi:(pi[1]-pi[0]*est_frac)/est_frac,zip(rgbp,rgbi)))
    
    print("colors:",rgbp,rgbi,clr)
    make_dummy_environment(ENVD,clr=clr,type=ENVIRONMENT_TYPE+1)
    print("Dummy environment created.")
