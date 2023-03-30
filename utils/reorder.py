import sys
"""All this script does is reorder the files in a dir, useful since after wholly extracting colmap results,
some images may be discarded due to being seemingly unrelated to the rest.
"""
if(len(sys.argv)==1):
    DIR=input("DIR=")
    EXT=input("EXT=")
    START=int(input("START="))
    PAD=int(input("PAD="))
else:
    DIR=sys.argv[1]
    EXT=sys.argv[2]
    START=int(sys.argv[3])
    PAD=int(sys.argv[4]) if len(sys.argv>4) else 0
import os
try:
    l = [str(i[:len(i)-len(EXT)]) for i in os.listdir(DIR) if i.endswith(EXT)]
    l.sort()
except:
    print("Names are not numbers; sorting lexicographically...")
    l = [str(i[:len(i)-len(EXT)]) for i in os.listdir(DIR) if i.endswith(EXT)]
    l.sort(key=int)
for i,n in enumerate(l):
    ii=format(START+i,f"0{PAD}d")
    os.rename(os.path.join(DIR,str(n)+EXT),os.path.join(DIR,ii+EXT))