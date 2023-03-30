import sdl2
from OpenGL import GL as gl
import struct
import numpy as np
from itertools import chain
import torch
NULL = gl.ctypes.c_void_p(0)

#TODO: make this function do nothing once you know OpenGL won't give you trouble
#NOTE: glGetError synchronizes with the GPU, which is quite slow.
def GL_CHECK_ERROR(err:str):
    err_id = gl.glGetError()
    if(err_id != gl.GL_NO_ERROR):
        raise RuntimeError(err)

class Renderer():
    def __init__(self,window_name:str="window name unset",w:int=500,h:int=500):
        self.views = {}
        WindowFlags = sdl2.SDL_WINDOW_OPENGL|sdl2.SDL_WINDOW_RESIZABLE|sdl2.SDL_WINDOW_SHOWN
        self.h = h;self.w = w
        self.window = sdl2.SDL_CreateWindow(bytes(window_name,"ascii"),50,50,self.w,self.h,WindowFlags)
        self.context = sdl2.SDL_GL_CreateContext(self.window)
        gl.glEnable(gl.GL_TEXTURE_2D)
        self.createView('main',0.0,0.0,1.0,1.0,0,False)
        fp = open("openGL_shaders/frag.glsl"  ).read()
        vp = open("openGL_shaders/vertex.glsl").read()
        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(vs, vp)
        gl.glCompileShader(vs)
        status=gl.ctypes.c_int()
        status=gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS)
        if (gl.GL_TRUE != status):
            infoLog=str(gl.glGetShaderInfoLog(vs))
            raise RuntimeError("VERTEX SHADER COMPILATION FAILED\n"+infoLog)
        gl.glShaderSource(fs, fp)
        gl.glCompileShader(fs)
        status=gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS)
        if (gl.GL_TRUE != status):
            infoLog=str(gl.glGetShaderInfoLog(fs))
            raise RuntimeError("SHADER SHADER COMPILATION FAILED\n"+infoLog)
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, vs)
        gl.glAttachShader(self.program, fs)
        gl.glLinkProgram(self.program)
        status=gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
        if (gl.GL_TRUE != status):
            infoLog=str(gl.glGetProgramInfoLog(self.program))
            raise RuntimeError("SHADER LINK FAILED\n"+ infoLog)

        self.buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        pos = [
         -0.5,-0.5,0.0,0.0,0.0,
          0.5,-0.5,0.0,1.0,0.0,
          0.5, 0.5,0.0,1.0,1.0,
         -0.5, 0.5,0.0,0.0,1.0,
        ]
        gl.glBufferData(gl.GL_ARRAY_BUFFER, struct.pack(len(pos)*'f',*pos), gl.GL_STATIC_DRAW)
        self.update()
        return
    def ensureWH(self,w,h):
        w=int(w);h=int(h)
        
        window_width = self.w; window_height = self.h
        if(w==window_width and h==window_height):
            return
        for v in self.views:
            self.views[v][1]=True
        self.ensureGL()
        sdl2.SDL_SetWindowSize(self.window, w, h)
        self.w=w;self.h=h
        for vid in self.views:
            if(self.views[vid][-1]):#if it is flexible
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.views[vid][0])
                GL_CHECK_ERROR("Texture could not be bound")

                #todo: delete these 2 lines and see if anything crashes when window is resized.
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                GL_CHECK_ERROR("Texture unpack alignment could not be changed")

                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, NULL)
                GL_CHECK_ERROR("Texture could not be resized")
                #TODO: update other parameters.

    def ensureGL(self):
        sdl2.SDL_GL_MakeCurrent(self.window,self.context)
    def update(self):
        self.ensureGL()
        w=gl.ctypes.c_int(0)
        h=gl.ctypes.c_int(0)
        wh=sdl2.SDL_GetWindowSize(self.window,w,h)
        w=w.value;h=h.value
        
        gl.glUseProgram(self.program)
        GL_CHECK_ERROR("Program could not be used")
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        GL_CHECK_ERROR("Buffer could not be bound")
        
        
        gl.glViewport(0, 0, w, h)
        GL_CHECK_ERROR("Viewport could not be set")
        gl.glClearColor(0., 1., 0., 1.)
        GL_CHECK_ERROR("Clear color could not be set")
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        GL_CHECK_ERROR("Screen could not be cleared")
        
        if (w != self.w or h != self.h):
            self.ensureWH(w, h)

        for view in self.views:
            #todo: set needs_update for all views on window resize.
            needs_update=self.views[view][1]
            if(needs_update):
                self.views[view][1]=False
                pos=self.views[view][2]
                old_coords=self.views[view][3]
                new_coords=(
                    int(pos[0]*w),
                    int(pos[1]*h),
                    int(pos[2]*w)-int(pos[0]*w),
                    int(pos[3]*h)-int(pos[1]*h),
                )
                new_size = (new_coords[2]-new_coords[0],new_coords[3]-new_coords[1])
                self.views[view][3]=new_coords
                pc=[*map(int,new_coords)]
                gl.glViewport(*pc)
                GL_CHECK_ERROR("Viewport could not be set")
                tex_id = self.views[view][0]
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
                GL_CHECK_ERROR("Texture could not be bound")
                #TODO:
                if(self.views[view][-1]):
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, new_size[0], new_size[1], 0, gl.GL_RGBA, gl.GL_BYTE, NULL)
                gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
                GL_CHECK_ERROR("Quad could not be drawn")
            else:
                coords=self.views[view][3]
                pc=[*map(int,coords)]
                gl.glViewport(*pc)
                GL_CHECK_ERROR("Viewport could not be set")
                tex_id = self.views[view][0]
                gl.glBindTexture(gl.GL_TEXTURE_2D,tex_id)
                GL_CHECK_ERROR("Texture could not be bound")
                gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
                GL_CHECK_ERROR("Quad could not be drawn")
        
        
        sdl2.SDL_GL_SwapWindow(self.window)
        
        events: list[sdl2.SDL_Event]=[]
        while(True):
            event: sdl2.SDL_Event= sdl2.SDL_Event()
            if(not sdl2.SDL_PollEvent(event)):break
            sdl2.SDL_FlushEvent(event.type)
            events.append(event)
        sdl2.SDL_PumpEvents(self.window)
        return events
    def upload_tensor(self,view_name:str,rgb:torch.Tensor):
        self.ensureGL()
        import io
        import torch
        #NOTE: Throws error" PyCUDA was compiled without GL extension support"
        #import pycuda.gl.autoinit
        #import pycuda.gl as cu_gl
        #TODO: find a way around this so you don't copy the data back and forth like an idiot
        view=self.views[view_name]
        tex_id = view[0]
        #print(rgb[::,::,:-1:].mean(),rgb[::,::,:-1:].max(),rgb[::,::,:-1:].min())
        rgb=rgb[::,::,:3:]#Only RGB is needed. channels such as depth may be discarded
        #No clue what the easiest way to do this is
        if(rgb.dtype==torch.float32):
            rgb = torch.clamp(rgb*255,0,255).to(dtype=torch.uint8)
        elif(rgb.dtype==torch.uint8):
            pass
        else:
            raise Exception(f"{rgb.dtype} data type does not have a protocol for being uploaded.")
        if(rgb.size(-1)>=3):
            rgb=rgb[::,::,:3:]
        elif(rgb.size(-1)==1):
            rgb=torch.cat([rgb,rgb,rgb],-1)
        else:
            raise Exception(f"{rgb.size(-1)} color channels do not have a protocol for being uploaded.")
        
        h,w,unused=rgb.size()
        
        rgb_cpu=rgb.cpu().reshape((rgb.numel(),)).contiguous()
        #This is by far the slowest thing.
        #rgb_bytes=bytearray(rgb_cpu)
        rgb_bytes=rgb_cpu.numpy().tobytes()
        
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        GL_CHECK_ERROR("GL_UNPACK_ALIGNMENT could not be set to 1")
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        GL_CHECK_ERROR("Texture bind failed")
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_bytes)
        GL_CHECK_ERROR("RGB tensor upload failed")
        
    def is_window_minimized(self):
        self.ensureGL()
        flags=sdl2.SDL_GetWindowFlags(self.window)
        flg=sdl2.SDL_WINDOW_MINIMIZED
        return 0!=(flags&flg)
    
    def sleep_until_not_minimized(self):
        self.ensureGL()
        while(self.is_window_minimized()):
            event=sdl2.SDL_Event()
            sdl2.SDL_WaitEvent(event)
        
    def upload_rgb(self,view_name:str,rgb):
        """WARNING: slow use for testing only
        Args:
            view_name (_type_): _description_
            rgb (_type_): _description_
        """
        rgb = np.asarray(rgb)
        w=len(rgb)
        h=len(rgb[0])
        c=len(rgb[0][0])
        t= gl.GL_RGBA8 if c==4 else gl.GL_RGB if c==3 else gl.GL_R if c==1 else []
        tex_id = self.views[view_name][0]
        data = bytearray(list(chain(rgb)))
        gl.glBindTexture(gl.GL_TEXTURE_2D,tex_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        GL_CHECK_ERROR("RGB upload failed")
    def removeView(self,name:str):
        tex_id = self.views[name][0]
        gl.glDeleteTextures([tex_id])
        del self.views[name]
    def createView(self,name:str,start_x:float,start_y:float,end_x:float,end_y:float,priori:int,flexible:bool=False):
        self.ensureGL()
        if(name in self.views):
            self.removeView(name)
        #TODO? custom class?
        tex_id = 0
        tex_id=gl.glGenTextures(1)
        
        GL_CHECK_ERROR("Opengl could not create a texture")

        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        GL_CHECK_ERROR("Opengl could not bind a texture")
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        GL_CHECK_ERROR("Opengl could not set a property of a texture")
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        GL_CHECK_ERROR("Opengl could not set a property of a texture")
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        GL_CHECK_ERROR("Opengl could not set a property of a texture")
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        GL_CHECK_ERROR("Opengl could not set a property of a texture")
        w=end_x-start_x
        h=end_y-start_y
        # I really need a dto for this.
        #                    id  needs_update   %-based coords     screen coords 
        self.views[name] = [tex_id,True,(start_x,start_y,end_x,end_y),(0,0,0,0),(w,h),priori,flexible]

        #Red is used as a sign something has not had data appropriately uploaded to it.
        data = bytearray( (255,0,0,255) )
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        GL_CHECK_ERROR("Opengl could not upload image data")