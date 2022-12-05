#version 140

in vec3 v_tex_coords;
out vec4 color;

uniform usamplerBuffer tex;

void main() {
    int buffer_index = int(floor(v_tex_coords[0] * 100) + floor(v_tex_coords[1] * 100) * 100);
    bool alive = bool(texelFetch(tex, buffer_index)[0]);
    color = alive ? vec4(1.0, 1.0, 1.0, 1.0) : vec4(0.1, 0.1, 0.1, 0.0);
}