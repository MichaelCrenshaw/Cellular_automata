#version 460 core

uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model;

in vec3 position;
in vec3 tex_coords;
out vec3 v_tex_coords;

void main() {
    v_tex_coords = tex_coords;

    gl_Position = vec4(position, 1.0);
}