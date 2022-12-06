#version 460 core

flat in uint patch_id;
out vec4 color;

uniform usamplerBuffer tex;

void main() {
    bool alive = bool(texelFetch(tex, int(patch_id)));
    color = alive ? vec4(float(patch_id), 1.0, 1.0, 1.0) : vec4(0.1, 0.1, 0.1, 0.0);
//    color = vec4(float(patch_id), 1.0, 1.0, 1.0);
}

//#version 460
//
//in vec3 v_tex_coords;
//out vec4 color;
//
//uniform usamplerBuffer tex;
//
//void main() {
//    uint dim1 = uint(floor(v_tex_coords[0] * 25u));
//    uint dim2 = uint(floor(v_tex_coords[1] * 25u)) * 25u;
//    uint dim3 = uint(floor(v_tex_coords[2] * 25u)) * 25u * 25u;
//    bool alive = bool(texelFetch(tex, int(dim1 + dim2 + dim3)));
//    color = alive ? vec4(1.0, 1.0, 1.0, 1.0) : vec4(0.1, 0.1, 0.1, 1.0);
//}