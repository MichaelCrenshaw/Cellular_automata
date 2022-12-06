#version 460 core

layout (vertices = 4) out;

uniform int tess_level = 25;
in vec3 v_tex_coords[];
in mat4 v_perspective[];
in mat4 model_view[];

out vec3 tex_coords[];
out mat4 e_perspective[];
out mat4 e_model_view[];


void main() {
    e_perspective[gl_InvocationID] = v_perspective[gl_InvocationID];
    e_model_view[gl_InvocationID] = model_view[gl_InvocationID];

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    tex_coords[gl_InvocationID] = v_tex_coords[gl_InvocationID];

    gl_TessLevelOuter[0] = tess_level;
    gl_TessLevelOuter[1] = tess_level;
    gl_TessLevelOuter[2] = tess_level;
    gl_TessLevelOuter[3] = tess_level;

    gl_TessLevelInner[0] = 0;
    gl_TessLevelInner[1] = 0;
}