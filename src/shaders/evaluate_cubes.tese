#version 460 core

layout (quads, equal_spacing, ccw) in;

in vec3 tex_coords[];
in mat4 e_perspective[];
in mat4 e_model_view[];

out vec4 patch_color;

uniform usamplerBuffer tex;

vec4 interpolate(in vec4 v0, in vec4 v1, in vec4 v2, in vec4 v3)
{
    vec4 a = mix(v0, v1, gl_TessCoord.x);
    vec4 b = mix(v3, v2, gl_TessCoord.x);
    return mix(a, b, gl_TessCoord.y);
}

void main() {
    vec3 patch_center = (tex_coords[0] + tex_coords[1]) / 2;
    vec3 pos_color = vec3(patch_center[0], patch_center[1], patch_center[2]);

    uint dim1 = uint(floor(tex_coords[gl_PrimitiveID][0] * 25u));
    uint dim2 = uint(floor(tex_coords[gl_PrimitiveID][1] * 25u)) * 25u;
    uint dim3 = uint(floor(tex_coords[gl_PrimitiveID][2] * 25u)) * 25u * 25u;
    patch_color = vec4(pos_color, bool(texelFetch(tex, int(dim1 + dim2 + dim3))) ? 1 : 0);

    gl_Position = interpolate(
    gl_in[0].gl_Position,
    gl_in[1].gl_Position,
    gl_in[2].gl_Position,
    gl_in[3].gl_Position);
}