#version 450 core
out vec2 v_tex;

const vec2 pos[4] = vec2[4](
    vec2(-1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(1.0, -1.0));

void main()
{
    v_tex = 0.5+0.5*vec2(pos[gl_VertexID].x,-pos[gl_VertexID].y);
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}