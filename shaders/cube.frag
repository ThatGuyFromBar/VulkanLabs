#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 mvp;
    mat4 model;
    vec4 ambientLight;
    vec4 lightColor;
    vec4 lightDir;
    vec4 cameraPos;
    vec4 specularColor;
    float shininess;
    float pad1, pad2, pad3;
} ubo;

void main() {
    vec3 n = normalize(fragNormal);
    vec3 l = normalize(-ubo.lightDir.xyz);
    vec3 v = normalize(ubo.cameraPos.xyz - fragPos);
    vec3 h = normalize(l + v);

    vec3 Kd = fragColor;                // Альбедо (цвет материала)
    vec3 Ks = ubo.specularColor.xyz;    // Цвет блика из ImGui

    // Модель Блинн-Фонга
    vec3 ambient = ubo.ambientLight.xyz * Kd;

    float diff = max(dot(n, l), 0.0);
    vec3 diffuse = diff * ubo.lightColor.xyz * Kd;

    float spec = pow(max(dot(n, h), 0.0), ubo.shininess);
    vec3 specular = spec * ubo.lightColor.xyz * Ks;

    vec3 result = ambient + diffuse + specular;
    outColor = vec4(result, 1.0);
}