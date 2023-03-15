#version 450 core
in vec2 v_tex;
uniform sampler2D texSampler;
out vec4 color;
void main()
{
    //color = vec4(1,0,0,1);
    color = texture(texSampler, v_tex);
}