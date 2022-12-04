#version 140

in vec3 position;
in vec2 tex_coords;
out vec2 v_tex_coords;

uniform mat4 perspective;
uniform mat4 transform_matrix;

void main() {
    v_tex_coords = tex_coords;
    gl_Position = perspective * transform_matrix * vec4(position, 1.0);
}