#TL;DR: cast rays perpendicular with x/y/z. Add to a list of points any intersection with any material data you want to keep (in my case albedo/color)
#Currently only works with .dae
SCALE=0.001
#NOTE: mode uses https://dl.acm.org/doi/10.1145/344779.344936
MODE_XYZ = "MODE_XYZ"
MODE_SPLIT = "MODE_SPLIT"
MODE = MODE_XYZ
PTH=input("PTH:")
FORMAT=input("FORMAT?(.bin/.txt)")
TYPE=".dae"#input("TYPE:")
OUTPUT = input("OUT:")
ENV = bool(int(input("env(0/1)?")))

import collada
import numpy as np
import math
from itertools import chain
#TODO: implement lol. I seriously have no idea what file format I should base this on    
positions = []
#Typically, RGB
other_channels = []
#https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
def crosses (p1,p2,p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
def PointInTriangle (pt,v1,v2,v3):
    d1 = crosses(pt, v1, v2)
    d2 = crosses(pt, v2, v3)
    d3 = crosses(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def round_down(x:float,multiple:float=1):
    return multiple*float(int(x/multiple))#NOTE: int(:float) rounds down
def round_up(x:float,multiple:float=1):
    return multiple*float(int(x/multiple))#NOTE: int(:float) rounds down


def add_triangle(tripos:tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]],
                 texture=None,tex_coords=None,vertex_colors=None,material_albedo=None,color=(1,1,1)):
    global positions,other_channels
    if(material_albedo!=None):color = material_albedo
    mins=list(map(min,zip(*tripos)))
    maxs=list(map(max,zip(*tripos)))
    clr_for_pos= lambda x,*a,**arg:color
    #TODO: other options?
    if(MODE==MODE_XYZ):
        #Method based on https://dl.acm.org/doi/10.1145/344779.344936
        for (SX,SY,SZ),(iSX,iSY,iSZ) in (((0,1,2),(0,1,2)),((1,2,0),(2,0,1)),((2,0,1),(1,2,0))):
            #would rounding up be faster?
            pos_ord = \
                       ((tripos[0][SX],tripos[0][SY],tripos[0][SZ]),
                        (tripos[1][SX],tripos[1][SY],tripos[0][SZ]),
                        (tripos[2][SX],tripos[2][SY],tripos[0][SZ]))
            (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) = pos_ord
            problem1 = np.array([[x1,y1,1],[x2,y2,1],[x3,y3,1]])
            problem2 = np.array([z1, z2, z3])
            try:
                #Solves:
                #x1*a+y1*b+c=z1
                #x2*a+y2*b+c=z2
                #x3*a+y3*b+c=z3
                #So we can find depths easily
                a,b,c = np.linalg.solve(problem1, problem2)
            except:
                continue#It is a perpendicular plane to this plane.
            for sample_x in np.arange(round_down(mins[SX],SCALE),maxs[SX],SCALE):
                for sample_y in np.arange(round_down(mins[SY],SCALE),maxs[SY],SCALE):
                    #now check if it is inside
                    is_inside = False
                    #TODO: redo? formula
                    is_inside = PointInTriangle((sample_x,sample_y),(x1,y1),(x2,y2),(x3,y3))
                    if(is_inside):
                        sample_z = a*sample_x+b*sample_y+c
                        
                        sample_x,sample_y,sample_z = [[sample_x,sample_y,sample_z][i] for i in (iSX,iSY,iSZ)]
                        positions.append((sample_x,sample_y,sample_z))
                        other_channels.append(clr_for_pos((sample_x,sample_y,sample_z)))#maybe add more stuff to the call?
    elif(MODE==MODE_SPLIT):#Method invented by me, probably not great
        max_diff = max(map(lambda mM:abs(mM[0]-mM[1]),zip(mins,maxs)))
        divs = max(1,math.ceil(max_diff/SCALE))
        for s in list(np.arange(0,1,1/(1+divs)))[1:]:
            for d in chain(np.arange(0,s,1/(1+divs)),list(map(lambda x:-1,np.arange(0,s,1/(1+divs))))[1:]):
                w1,w2=(s-d)/2,(s+d)/2
                w3=1-w1-w2
                if(w3<0):
                    sample_x,sample_y,sample_z = map(lambda x123:x123[0]*w1+x123[1]*w2+x123[2]*w3,zip(*tripos))
                    positions.append((sample_x,sample_y,sample_z))
                    other_channels.append(clr_for_pos((sample_x,sample_y,sample_z)))
        

if(TYPE == ".dae"):
    data =collada.Collada(PTH)
    geometries = list(data.scene.objects('geometry'))
    for idx,g in enumerate(geometries):
        geometry:collada.geometry.BoundGeometry=g
        
        print(f"Processing geometry {idx+1}/{len(geometries)}")
        transform=geometry.matrix
        for primitive in geometry.primitives():
            p:collada.geometry.primitive.Primitive=primitive
            if(type(p) == collada.geometry.triangleset.BoundTriangleSet):
                triangle_set:collada.geometry.triangleset.BoundTriangleSet=p
                mat:str=triangle_set.material
                for triangle in triangle_set:
                    p0,p1,p2 = triangle.vertices
                    #These are the triangle positions
                    add_triangle((p0,p1,p2),color=(0.5,0.5,0.5))
            else:
                print(str(p) + " could not be transformed because it is not a triangleset.")

def writePointCloudBin():
    import struct
    print(f'writing point cloud to {OUTPUT+"/points.bin"}')
    num_channels = len(other_channels[0])
    w=len(positions[0])+num_channels
    
    with open(OUTPUT+"/points.bin","wb") as f:
        f.write(struct.pack("qq",1,w))
        for p,c in zip(positions,other_channels):
            f.write(struct.pack("f"*w,*chain(p,c)))

    print(f'writing environment to {OUTPUT+"/environment.bin"}')
    with open(OUTPUT+"/environment.bin","wb") as f:
        env_res = 10
        f.write(struct.pack("qqqqq",4,6,env_res,env_res,num_channels+1))
        for _face in range(6):
            for _X in range(env_res):
                for _Y in range(env_res):
                    f.write(struct.pack("f"*(num_channels)+"f",*([0]*num_channels+[-1])))

print(f"Extracted {len(positions)} points")
if(FORMAT==".bin"):
    writePointCloudBin()
print("Done.")