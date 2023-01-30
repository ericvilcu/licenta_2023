#include "Renderer.hpp"

#if WIN32
#include <Windows.h>
#endif
#include <iostream>
//#include <GL/gl3w.h>
#include "HeaderThatSupressesWarnings.h"
#include <SDL.h>
#include <SDL_rect.h>
#include <SDL_opengl.h>
#include <gl/gl.h>
#include <gl/glu.h>
#include "HeaderThatReenablesWarnings.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>


typedef int32_t i32;
typedef uint32_t u32;
typedef int32_t b32;
typedef unsigned char u8;

constexpr auto WinWidth = 200;
constexpr auto WinHeight = 100;

//todo? make macro throw an exception instead?
//Note: glGetError syncs with the GPU every time it is called. That is very slow.
#if _DEBUG
#define NOGLERROR(s) do{auto err = glGetError();if(err != GL_NO_ERROR){std::cerr<<s<<'\n'<<"at line "<<__LINE__<<" file "<<__FILE__<<"\ncode:"<<err<<" (hex)"<<std::hex<<err<<'\n';exit(-1);}}while(false)
#define NOTNULL(X) do{if(X==NULL){std::cout<< #X<<" has value "<<X<<" at line "<<__LINE__<<'\n';exit(-1);}}while(false)
#else // _DEBUG
#define NOGLERROR(s)
#define NOTNULL(X)
#endif

const char vertex[]{ 
"#version 450 core\n\
out vec2 v_tex;\n\
\n\
const vec2 pos[4] = vec2[4](\n\
    vec2(-1.0, 1.0),\n\
    vec2(-1.0, -1.0),\n\
    vec2(1.0, 1.0),\n\
    vec2(1.0, -1.0));\n\
\n\
void main()\n\
{\n\
    v_tex = 0.5+0.5*vec2(pos[gl_VertexID].x,-pos[gl_VertexID].y);\n\
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);\n\
}\n" };
const char frag[]{
"#version 450 core\n\
in vec2 v_tex;\n\
uniform sampler2D texSampler;\n\
out vec4 color;\n\
void main()\n\
{\n\
    color = texture(texSampler, v_tex);\n\
}\n" };

Renderer::Renderer(const std::string& window_name)
{
    int status;
    u32 WindowFlags = SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE|SDL_WINDOW_SHOWN;
    window = SDL_CreateWindow(window_name.c_str(), 50, 50, WinWidth, WinHeight, WindowFlags);
    assert(("SDL+OpenGL wondow cpuld not be initialized.", window));
    context = SDL_GL_CreateContext((SDL_Window*)window);
    glEnable(GL_TEXTURE_2D);
    createView(ViewTypeEnum::MAIN_VIEW,0.0f,1.0f,0.0f,1.0f,0);

    const auto glCreateShader  = (PFNGLCREATESHADERPROC )SDL_GL_GetProcAddress("glCreateShader" );
    NOTNULL(glCreateShader);
    const auto glShaderSource  = (PFNGLSHADERSOURCEPROC )SDL_GL_GetProcAddress("glShaderSource" );
    NOTNULL(glShaderSource);
    const auto glCompileShader = (PFNGLCOMPILESHADERPROC)SDL_GL_GetProcAddress("glCompileShader");
    NOTNULL(glCompileShader);
    const auto glGetShaderiv   = (PFNGLGETSHADERIVPROC  )SDL_GL_GetProcAddress("glGetShaderiv"  );
    NOTNULL(glGetShaderiv);
    const auto glDeleteShader  = (PFNGLDELETESHADERPROC )SDL_GL_GetProcAddress("glDeleteShader" );
    NOTNULL(glDeleteShader);
    const auto glCreateProgram = (PFNGLCREATEPROGRAMPROC)SDL_GL_GetProcAddress("glCreateProgram");
    NOTNULL(glCreateProgram);
    const auto glAttachShader  = (PFNGLATTACHSHADERPROC )SDL_GL_GetProcAddress("glAttachShader" );
    NOTNULL(glAttachShader);
    const auto glLinkProgram   = (PFNGLLINKPROGRAMPROC  )SDL_GL_GetProcAddress("glLinkProgram"  );
    NOTNULL(glLinkProgram);
    const auto glGetProgramiv  = (PFNGLGETPROGRAMIVPROC )SDL_GL_GetProcAddress("glGetProgramiv" );
    NOTNULL(glGetProgramiv);
    const auto glDeleteProgram = (PFNGLDELETEPROGRAMPROC)SDL_GL_GetProcAddress("glDeleteProgram");
    NOTNULL(glDeleteProgram);
    //std::cerr << glGetString(GL_VERSION)<<'\n';
    //Create & compile shaders
    //SDL_gl
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char* vp = &vertex[0]; const char* fp = &frag[0];

    glShaderSource(vs, 1, &vp, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
    if (GL_TRUE != status)
    {
        const auto glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)SDL_GL_GetProcAddress("glGetShaderInfoLog");
        char infoLog[512];
        glGetShaderInfoLog(vs, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        assert(false);
    }
    glShaderSource(fs, 1, &fp, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
    if (GL_TRUE != status)
    {
        const auto glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)SDL_GL_GetProcAddress("glGetShaderInfoLog");
        char infoLog[512];
        glGetShaderInfoLog(fs, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        assert(false);
    }
    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (GL_TRUE != status)
    {
        const auto glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)SDL_GL_GetProcAddress("glGetProgramInfoLog");
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
        assert(false);
    }

    //Bind a buffer for positions/texture coordonates.
    const auto glBindBuffer = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    NOTNULL(glBindBuffer);
    const auto glGenBuffers = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    NOTNULL(glGenBuffers);
    const auto glBufferData = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    NOTNULL(glBufferData);
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    float pos[] = {
        -0.5,-0.5,0.0,0.0,0.0,
         0.5,-0.5,0.0,1.0,0.0,
         0.5, 0.5,0.0,1.0,1.0,
        -0.5, 0.5,0.0,0.0,1.0,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(pos), pos, GL_STATIC_DRAW);
    update();/**/
}

void Renderer::removeView(Renderer::ViewType vt){
    glDeleteTextures(1, &views[vt]->textureId);
    NOGLERROR("Opengl could not delete a texture");
    views.erase(vt);
}

void Renderer::createView(Renderer::ViewType vt, float lp, float rp, float up, float dp, int priority, bool resize_on_window_resize) {
#ifdef _DEBUG
    assert(lp<rp);
    assert(up<dp);
#endif // _DEBUG
    if(views.count(vt)){
        removeView(vt);
    }
    views[vt] = std::make_unique<View>();
    Renderer::View& view = *views[vt];
    view.needs_update = true;
    view.lp=lp;
    view.rp=rp;
    view.up=up;
    view.dp=dp;
    view.priority = priority;
    view.resize_on_window_resize = resize_on_window_resize;
    glGenTextures(1,&view.textureId);
    NOGLERROR("Opengl could not create a texture");

    glBindTexture(GL_TEXTURE_2D, view.textureId);
    NOGLERROR("Opengl could not bind a texture");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    NOGLERROR("Opengl could not set a property of a texture");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    NOGLERROR("Opengl could not set a property of a texture");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    NOGLERROR("Opengl could not set a property of a texture");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    NOGLERROR("Opengl could not set a property of a texture");

    //Red is used as a sign something has not had data appropriately uploaded to it.
    u8 data[] = { 255,0,0,255 };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    NOGLERROR("Opengl could not upload image data");

}

void Renderer::uploadRGBA8(Renderer::ViewType vt, void* data, int width, int height)
{
    glBindTexture(GL_TEXTURE_2D, views[vt]->textureId);
    NOGLERROR("texture could not be bound.");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    NOGLERROR("texture could not be updated.");
}

void Renderer::update() {
    //These calls are windows-specific
    const auto glBindBuffer = (PFNGLBINDBUFFERPROC   )SDL_GL_GetProcAddress("glBindBuffer");
    NOTNULL(glBindBuffer);
    const auto glUseProgram = (PFNGLUSEPROGRAMPROC   )SDL_GL_GetProcAddress("glUseProgram");
    NOTNULL(glUseProgram);
    SDL_GL_MakeCurrent((SDL_Window*)window, context);
    int w, h;
    SDL_GetWindowSize((SDL_Window*)window, &w, &h);
    

    glViewport(0, 0, w, h);
    NOGLERROR("Viewport could not be set");
    glClearColor(0.f, 0.f, 0.f, 1.f);
    NOGLERROR("Clear color could not be set");
    glClear(GL_COLOR_BUFFER_BIT);
    NOGLERROR("Screen could not be cleared");

    //Todo: scale image to keep aspect ratio
    glUseProgram(program);
    NOGLERROR("Program could not be used");
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    NOGLERROR("Buffer could not be bound");
    
    for(auto& v:views){
        //todo: set needs_update for all views on window resize.
        if(v.second->needs_update){
            v.second->needs_update = false;
            v.second->left=(int)(v.second->lp * (float)w);
            v.second->top=(int)(v.second->up * (float)h);
            v.second->width=(int)(v.second->rp * (float)w) - v.second->left;
            v.second->height=(int)(v.second->dp * (float)h) - v.second->top;
            glViewport(v.second->left, v.second->top, v.second->width, v.second->height);
            NOGLERROR("Viewport could not be set");
            glBindTexture(GL_TEXTURE_2D, v.second->textureId);
            NOGLERROR("Texture could not be bound");
            if(v.second->resize_on_window_resize)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, v.second->width, v.second->height, 0, GL_RGBA, GL_BYTE, NULL);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            NOGLERROR("Quad could not be drawn");
        }
        else {
            glViewport(v.second->left, v.second->top, v.second->width, v.second->height);
            NOGLERROR("Viewport could not be set");
            glBindTexture(GL_TEXTURE_2D, v.second->textureId);
            NOGLERROR("Texture could not be bound");
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            NOGLERROR("Quad could not be drawn");
        }
    }

    SDL_GL_SwapWindow((SDL_Window*)window);

    if (w != window_width || h != window_height) {
        ensureWH(w, h);
    }
}

Renderer::~Renderer()
{
	SDL_Quit();
}

bool Renderer::shouldClose()
{
    return SDL_HasEvent(SDL_QUIT);
}

void Renderer::ensureWH(int width, int height)
{
    for (auto& v : views) {
        v.second->needs_update = true;
    }
    window_width = width; window_height = height;
    SDL_GL_MakeCurrent((SDL_Window*)window, context);
    SDL_SetWindowSize((SDL_Window*)window, width, height);
    glBindTexture(GL_TEXTURE_2D, views[ViewTypeEnum::MAIN_VIEW]->textureId);
    NOGLERROR("Texture could not be bound");

    //todo: delete these 2 lines and see if anything crashes when window is resized.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    NOGLERROR("Texture unpack alignment could not be changed");

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    NOGLERROR("Texture could not be resized");
}

void Renderer::getWH(int& width, int& height) const
{
    SDL_GetWindowSize((SDL_Window*)window, &width, &height);
}
void Renderer::getWH(int* widthp, int* heightp) const
{
    SDL_GetWindowSize((SDL_Window*)window, widthp, heightp);
}
