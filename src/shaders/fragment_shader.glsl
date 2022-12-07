#version 460 core

flat in vec4 patch_color;

out vec4 color;

uniform usamplerBuffer tex;

void main() {
    color = patch_color;
}