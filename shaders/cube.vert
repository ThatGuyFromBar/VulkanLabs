#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 fragColor;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 mvp;
} ubo;

void main() {
    fragColor = inColor;
    gl_Position = ubo.mvp * vec4(inPosition, 1.0);
}