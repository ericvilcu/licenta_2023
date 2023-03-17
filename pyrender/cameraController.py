import math
import sdl2
import sdl2.events as sdl_events
from itertools import chain
def clamp(x,mn,mx):
    return min(mx,max(x,mn))
class CameraController():
    def __init__(self) -> None:
        self.key_state=[0]*8
        self.mouse_state=[0]*3
        self.CW,self.CH=300.,300.
        self.needs_rebuild=True
        self.camera=[0.]
        self.ypr=[0.]*3
        self.fov=90
        self.position=[0.]*3
    
    def camera_type(self):
        return 0#Only pinhole projection for now.
    @staticmethod
    def rotation_from(yaw:float,pitch:float,roll:float):
        sin,cos=math.sin,math.cos
        y,p,r=yaw,pitch,0
        rotation=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        #uses the same method used at http://msl.cs.uiuc.edu/planning/node102.html , just multiplying several matrices together.
        
        #If multiplying to the right
        rotation[0][0] = float(cos(y))
        rotation[0][1] = float(0)
        rotation[0][2] = float(-sin(y))
        rotation[1][0] = float(sin(p) *-sin(y))
        rotation[1][1] = float(cos(p))
        rotation[1][2] = float(-sin(p)* cos(y))
        rotation[2][0] = float(cos(p) * sin(y))
        rotation[2][1] = float(sin(p))
        rotation[2][2] = float(cos(p) * cos(y))
        
        #If multiplying to the left
        # rotation[0][0] = float(cos(y))
        # rotation[0][1] = float(sin(y) * sin(p))
        # rotation[0][2] = float(sin(y) * cos(p))
        # rotation[1][0] = float(0)
        # rotation[1][1] = float(cos(p))
        # rotation[1][2] = float(-sin(p))
        # rotation[2][0] = float(-sin(y))
        # rotation[2][1] = float(sin(p) * cos(y))
        # rotation[2][2] = float(cos(y) * cos(p))
        return rotation
    
    def rebuild_camera(self):
        self.needs_rebuild=False
        rotation:chain[float] = chain(*self.rotation_from(*self.ypr))
        af=math.atan(self.fov/180*math.pi)
        self.camera=[0.,0.,self.CW,self.CH,*rotation,*self.position,self.CW/2,self.CH/2,self.CW/2,self.CH/2]#*(2+9+3+4)
    
    def multiply_with_rotation():
        pass
    
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
                    sensitivity=1e-3
                    self.ypr[0]=(self.ypr[0]-event.motion.xrel*sensitivity)%(2*math.pi)
                    self.ypr[1]=clamp(self.ypr[1]-event.motion.yrel*sensitivity,-math.pi*0.9,math.pi*0.9)
                    self.needs_rebuild=True
            elif(event.type==sdl_events.SDL_KEYDOWN or event.type==sdl_events.SDL_KEYUP):
                keysym:sdl_events.SDL_Keysym=event.key.keysym
                is_pressed=int(bool(event.type==sdl_events.SDL_KEYDOWN))
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
                elif(keysym.sym==sdl2.SDLK_e):
                    self.key_state[7]=is_pressed
            else:
                pass
        
        motion=(self.key_state[0]-self.key_state[1],self.key_state[2]-self.key_state[3])
        if(motion!=(0,0)):
            if(self.needs_rebuild):
                self.rebuild_camera()
            rot=self.camera[4:4+9]
            motion=(delta*-motion[1],0.,delta*-motion[0])
            
            motion: tuple[float,float,float]=tuple(
                sum(motion[i]*rot[j+3*i] for i in range(3)) for j in range(3)
            )
            for i in range(3):
                self.position[i]+=motion[i]
            
            self.needs_rebuild=True
        
        
        
        
    def get_camera(self):
        if(self.needs_rebuild):
            self.rebuild_camera()
        return self.camera.copy()
    
    