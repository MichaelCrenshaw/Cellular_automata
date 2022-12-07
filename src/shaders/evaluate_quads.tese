#version 460 core

layout (quads, equal_spacing, ccw) in;

in vec3 tex_coords[];
in mat4 e_perspective[];
in mat4 e_model_view[];

out uint patch_id;

vec4 interpolate(in vec4 v0, in vec4 v1, in vec4 v2, in vec4 v3)
{
    vec4 a = mix(v0, v1, gl_TessCoord.x);
    vec4 b = mix(v3, v2, gl_TessCoord.x);
    return mix(a, b, gl_TessCoord.y);
}

void main() {
    patch_id = uint(gl_PrimitiveID);

//    vec4 pos0 = gl_in[0].gl_Position;
//    vec4 pos1 = gl_in[1].gl_Position;
//    vec4 pos2 = gl_in[2].gl_Position;
//    vec4 pos3 = gl_in[3].gl_Position;
//
//    vec4 leftPos = pos0 + gl_TessCoord.y * (pos3 - pos0);
//    vec4 rightPos = pos1 + gl_TessCoord.y * (pos2 - pos1);
//    vec4 pos = leftPos + gl_TessCoord.x * (rightPos - leftPos);
//
//    gl_Position = pos * e_model_view[patch_id] * e_perspective[patch_id];

    gl_Position = interpolate(
    gl_in[0].gl_Position,
    gl_in[1].gl_Position,
    gl_in[2].gl_Position,
    gl_in[3].gl_Position);
}