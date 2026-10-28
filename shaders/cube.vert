#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 mvp;
    mat4 model;
} ubo;

void main() {
    // Мировые координаты вершины
    fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
    
    // Перевод нормали в мировые координаты с помощью матрицы модели (обратная транспонированная)
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    
    fragColor = inColor;
    gl_Position = ubo.mvp * vec4(inPosition, 1.0);
}