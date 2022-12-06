#version 140

in vec3 position;
in vec3 tex_coords;
out vec3 v_tex_coords;

uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model;

void main() {
    v_tex_coords = tex_coords;

    mat4 model_view = view * model;
    gl_Position = perspective * model_view * vec4(position, 1.0);
}