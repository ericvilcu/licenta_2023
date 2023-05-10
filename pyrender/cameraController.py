import math
import sdl2
import sdl2.events as sdl_events
from itertools import chain
import torch
def clamp(x,mn,mx):
    return min(mx,max(x,mn))
class CameraController():
    def __init__(self) -> None:
        self.use_neural=False
        self.key_state=[0]*8
        self.mouse_state=[0]*3
        self.CW,self.CH=1241.0, 376.0#300.,300.
        self.needs_rebuild=True
        self.camera=[0.]
        self.ypr=[0.]*3
        self.position=[0.]*3
        self.flip_x=True
        self.shown_dim=0
        self.shown_ndim=3
        self.take_screencap=False
        self.samples_enabled=True
    
    def camera_type(self):
        return 0#Only pinhole projection for now.
    @staticmethod
    def rotation_from(yaw:float,pitch:float,roll:float,flip_x:bool):
        sin,cos=math.sin,math.cos
        y,p,r=yaw,pitch,0
        rotation=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        #uses the same method used at http://msl.cs.uiuc.edu/planning/node102.html , just multiplying several matrices together.
        
        #If multiplying to the right
        # rotation[0][0] = float(cos(y))
        # rotation[0][1] = float(0)
        # rotation[0][2] = float(-sin(y))
        # rotation[1][0] = float(sin(p) *-sin(y))
        # rotation[1][1] = float(cos(p))
        # rotation[1][2] = float(-sin(p)* cos(y))
        # rotation[2][0] = float(cos(p) * sin(y))
        # rotation[2][1] = float(sin(p))
        # rotation[2][2] = float(cos(p) * cos(y))
        xm=-1 if flip_x else 1
        #If multiplying to the left
        rotation[0][0] = float(cos(y))*xm
        rotation[0][1] = float(sin(y) * sin(p))*xm
        rotation[0][2] = float(sin(y) * cos(p))*xm
        rotation[1][0] = float(0)
        rotation[1][1] = float(cos(p))
        rotation[1][2] = float(-sin(p))
        rotation[2][0] = float(-sin(y))
        rotation[2][1] = float(sin(p) * cos(y))
        rotation[2][2] = float(cos(y) * cos(p))
        return rotation
    
    def rebuild_camera(self):
        self.needs_rebuild=False
        rotation:chain[float] = chain(*self.rotation_from(*self.ypr,flip_x=self.flip_x))
        #TODO: add shortcuts for controlling lightness?
        self.camera=[0.,0.,self.CW,self.CH,1,*rotation,*self.position,self.CW/2,self.CH/2,700.,700.]#self.CW/2,self.CH/2]#*(2+9+3+4)
    
    def only_shown_dimensions(self,tsr:torch.Tensor):
        ch=tsr.size(-1)
        return torch.stack([tsr[:,:,((i+self.shown_dim)%ch+ch)%ch] for i in range(self.shown_ndim)],-1)
    
    def process_sdl_events(self,events:list[sdl_events.SDL_Event],delta:float=1/60):
        for event in events:#These if chains are quite slow. note that this is going to be basically called every frame.
            if(event.type==sdl_events.SDL_MOUSEBUTTONDOWN):
                if(event.button.button==sdl2.SDL_BUTTON_LEFT):
                    self.mouse_state[0]=1
            elif(event.type==sdl_events.SDL_MOUSEBUTTONUP):
                if(event.button.button==sdl2.SDL_BUTTON_LEFT):
                    self.mouse_state[0]=0
            elif(event.type==sdl_events.SDL_MOUSEMOTION):
                if(self.mouse_state[0]==1):
                    sensitivity=5e-3
                    self.ypr[0]=(self.ypr[0]-event.motion.xrel*sensitivity)%(2*math.pi)
                    self.ypr[1]=clamp(self.ypr[1]+event.motion.yrel*sensitivity,-math.pi*0.49,math.pi*0.49)
                    self.needs_rebuild=True
            elif(event.type==sdl_events.SDL_KEYDOWN or event.type==sdl_events.SDL_KEYUP):
                keysym:sdl_events.SDL_Keysym=event.key.keysym
                is_pressed=int(bool(event.type==sdl_events.SDL_KEYDOWN))
                #This is not great, but since this is python and match/case does the same thing under the hood, what am I supposed to do? a lambda dictionary?
                if(keysym.sym==sdl2.SDLK_w):
                    self.key_state[0]=is_pressed
                elif(keysym.sym==sdl2.SDLK_s):
                    self.key_state[1]=is_pressed
                elif(keysym.sym==sdl2.SDLK_a):
                    self.key_state[2]=is_pressed
                elif(keysym.sym==sdl2.SDLK_d):
                    self.key_state[3]=is_pressed
                elif(keysym.sym==sdl2.SDLK_q):
                    self.key_state[4]=is_pressed
                elif(keysym.sym==sdl2.SDLK_e):
                    self.key_state[5]=is_pressed
                elif(keysym.sym==sdl2.SDLK_LSHIFT):
                    self.key_state[6]=is_pressed
                elif(keysym.sym==sdl2.SDLK_r):
                    self.key_state[7]=is_pressed
                elif(keysym.sym==sdl2.SDLK_p):
                    if(is_pressed):
                        self.take_screencap=True
                elif(keysym.sym==sdl2.SDLK_n and not is_pressed):
                    self.use_neural=not self.use_neural
                elif(keysym.sym==sdl2.SDLK_f and not is_pressed):
                    self.flip_x=not self.flip_x
                elif(keysym.sym==sdl2.SDLK_KP_PLUS and not is_pressed):
                    self.shown_dim+=1
                elif(keysym.sym==sdl2.SDLK_KP_MINUS and not is_pressed):
                    self.shown_dim-=1
                elif(keysym.sym==sdl2.SDLK_KP_MULTIPLY and not is_pressed):
                    self.shown_ndim=4-self.shown_ndim
                elif(keysym.sym==sdl2.SDLK_F5 and not is_pressed):
                    self.samples_enabled=not self.samples_enabled
            else:
                pass
        
        motion=(self.key_state[0]-self.key_state[1],self.key_state[2]-self.key_state[3])
        if(motion!=(0,0)):
            if(self.needs_rebuild):
                self.rebuild_camera()
            rot=self.camera[5:5+9]
            motion=(delta*-motion[1],0.,delta*-motion[0])
            
            motion: tuple[float,float,float]=tuple(
                sum(motion[i]*rot[i+3*j] for i in range(3)) for j in range(3)
            )
            for i in range(3):
                self.position[i]+=motion[i]
            
            self.needs_rebuild=True
        
        
        
        
    def get_camera(self):
        if(self.needs_rebuild):
            self.rebuild_camera()
        return self.camera.copy()
    
    