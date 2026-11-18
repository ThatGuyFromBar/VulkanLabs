#include "application.hpp"

#include <imgui.h>

#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

#include "graphics_internal.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace application {

// Глобальные переменные
// Буфер вершин
static VkBuffer s_vertexBuffer = VK_NULL_HANDLE;
static VmaAllocation s_vertexAlloc = {};
// Буфер индексов
static VkBuffer s_indexBuffer = VK_NULL_HANDLE;
static VmaAllocation s_indexAlloc = {};
static uint32_t s_indexCount = 0;

// Структура для текстуры
struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkSampler sampler = VK_NULL_HANDLE;
};

// Глобальные переменные текстур
static Texture s_textureCube;
static Texture s_textureFloor;

// Ресурсы Uniform Buffer и Descriptors
struct GlobalUniforms {
    alignas(16) float mvp[16];          
    alignas(16) float model[16];        
    alignas(16) float ambientLight[4];  // Сначала ambient
    alignas(16) float lightColor[4];    // Потом цвет
    alignas(16) float lightDir[4];      // Потом направление
    alignas(16) float cameraPos[4];     // Потом камера
	alignas(16) float specularColor[4]; // Цвет блика
    float shininess;                    
    float pad1, pad2, pad3;             
};
static VkBuffer s_uniformBuffer = VK_NULL_HANDLE;
static VmaAllocation s_uniformAlloc = {};
static void* s_uniformBufferMapped = nullptr; // Отображенная память UBO

// второй UBO для пола
static VkBuffer s_uniformBufferFloor = VK_NULL_HANDLE;
static VmaAllocation s_uniformAllocFloor = {};
static void* s_uniformBufferMappedFloor = nullptr;

// Выравнивание для динамического UBO
static VkDeviceSize s_dynamicAlignment = 0;
static VkDeviceSize s_modelUniformsSize = 0;

static VkDescriptorSetLayout s_descriptorSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool s_descriptorPool = VK_NULL_HANDLE;
static VkDescriptorSet s_descriptorSet = VK_NULL_HANDLE;      // для куба
static VkDescriptorSet s_descriptorSetFloor = VK_NULL_HANDLE; // для пола

// границы индексов
static uint32_t s_cubeIndexCount   = 0;
static uint32_t s_floorIndexOffset = 0;
static uint32_t s_floorIndexCount  = 0;

static VkPipeline s_pipeline = VK_NULL_HANDLE; // конвейер
static VkPipelineLayout s_pipelineLayout = VK_NULL_HANDLE; // макет конвейера
static VkShaderModule s_vertModule = VK_NULL_HANDLE; // вершинный шейдер
static VkShaderModule s_fragModule = VK_NULL_HANDLE; // фрагментный шейдер

// UI пользователя
static bool s_usePerspective = true; // смена вида
static float s_rotation[3] = {0.0f, 0.0f, 0.0f}; // поворот куба
static bool s_autoRotate = false; // автовращение
// lab2
static float s_camPos[3] = {0.0f, 0.0f, 3.5f};
static float s_lightDir[3] = {0.0f, -1.0f, -1.0f};
static float s_lightColor[3] = {1.0f, 1.0f, 1.0f};
static float s_ambientLight[3] = {0.2f, 0.2f, 0.2f};
static float s_shininess = 32.0f;
static float s_camRot[3] = {0.0f, 0.0f, 0.0f}; // Поворот камеры (Pitch, Yaw, Roll)
static float s_fov = 45.0f;                    // Угол обзора (FOV)
static float s_specularColor[3] = {1.0f, 1.0f, 1.0f}; // Цвет блика (Ks)

static bool createTexture(const char* path, Texture& outTexture) {
    using namespace graphics::internal;
    auto& ctx = context;

    // Загружаем пиксели
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "Failed to load texture image: " << path << '\n';
        return false;
    }
    VkDeviceSize imageSize = texWidth * texHeight * 4; // RGBA8

    // Создаем staging буфер
    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingInfo.size = imageSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VmaAllocationInfo stagingResult;
    if (vmaCreateBuffer(ctx.allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAlloc, &stagingResult) != VK_SUCCESS) {
        stbi_image_free(pixels);
        return false;
    }

    std::memcpy(stagingResult.pMappedData, pixels, imageSize);
    stbi_image_free(pixels);

    // Создаем VkImage в VRAM
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = texWidth;
    imageInfo.extent.height = texHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo imgAllocInfo{};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(ctx.allocator, &imageInfo, &imgAllocInfo, &outTexture.image, &outTexture.alloc, nullptr) != VK_SUCCESS) {
        vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);
        return false;
    }

    // Командный буфер
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = ctx.graphics_queue_index;
    VkCommandPool tempPool;
    vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &tempPool);

    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = tempPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Барьер из UNDEFINED в TRANSFER_DST
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = outTexture.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Копирование
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {(uint32_t)texWidth, (uint32_t)texHeight, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, outTexture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Барьер из TRANSFER_DST в SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    // Отправка и ожидание
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence);
    vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, fence);
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(ctx.device, fence, nullptr);
    vkDestroyCommandPool(ctx.device, tempPool, nullptr);
    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);

    // Создаем VkImageView
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = outTexture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(ctx.device, &viewInfo, nullptr, &outTexture.view);

    // Создаем VkSampler
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkCreateSampler(ctx.device, &samplerInfo, nullptr, &outTexture.sampler);

    return true;
}

bool initialize() {
	using namespace graphics::internal;
	auto& ctx = context;

	// Загружаем текстуры
	if (!createTexture("textures/cube.png", s_textureCube)) return false;
	if (!createTexture("textures/floor.png", s_textureFloor)) return false;

	// читаем бинарник шейдера .spv
	auto readFile = [](const char* path) -> std::vector<char> {
		std::ifstream ifs(path, std::ios::binary | std::ios::ate);
		if (!ifs) return {};
		size_t size = (size_t)ifs.tellg();
		std::vector<char> data(size);
		ifs.seekg(0);
		ifs.read(data.data(), (std::streamsize)size);
		return data;
	};

	// создаем модуль шейдера
	auto createShaderModule = [&](const char* path, VkShaderModule& outModule) -> bool {
		auto data = readFile(path);
		if (data.empty()) {
			std::cerr << "Failed to read shader file: " << path << '\n';
			return false;
		}

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = data.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(data.data());

		if (vkCreateShaderModule(ctx.device, &createInfo, nullptr, &outModule) != VK_SUCCESS) {
			std::cerr << "Failed to create shader module: " << path << '\n';
			return false;
		}
		return true;
	};

	if (!createShaderModule("shaders/cube.vert.spv", s_vertModule)) return false;
	if (!createShaderModule("shaders/cube.frag.spv", s_fragModule)) return false;

	// ЗАДАЕМ КУБ (позиция и цвет, нормали сгенерируем процедурно)
	struct Vertex { 
		float pos[3];    // Позиция
		float normal[3]; // Нормаль (будет заполнена процедурно)
		float uv[2];  // Текстурные координаты
	};

	std::vector<Vertex> vertices = {
		// 1. Передняя грань (Z+)
		{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}}, // UV: левый низ
		{{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f}}, // UV: правый низ
		{{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}}, // UV: правый верх
		{{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}}, // UV: левый верх

		// 2. Задняя грань (Z-)
		{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
		{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
		{{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		{{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

		// 3. Левая грань (X-)
		{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
		{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
		{{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		{{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

		// 4. Правая грань (X+)
		{{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
		{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
		{{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		{{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

		// 5. Нижняя грань (Y-)
		{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
		{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
		{{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

		// 6. Верхняя грань (Y+)
		{{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
		{{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
		{{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
		{{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
	};

	// Индексы для 6 граней (36 индексов)
	std::vector<uint16_t> indices = {
		0,  1,  2,   2,  3,  0,   // Передняя
		4,  5,  6,   6,  7,  4,   // Задняя
		8,  9,  10,  10, 11, 8,   // Левая
		12, 13, 14,  14, 15, 12,  // Правая
		16, 17, 18,  18, 19, 16,  // Нижняя
		20, 21, 22,  22, 23, 20    // Верхняя
	};

	// ПРОЦЕДУРНАЯ ГЕНЕРАЦИЯ НОРМАЛЕЙ
	for (size_t i = 0; i < indices.size(); i += 3) {
		uint16_t idx0 = indices[i + 0];
		uint16_t idx1 = indices[i + 1];
		uint16_t idx2 = indices[i + 2];

		Vertex& v0 = vertices[idx0];
		Vertex& v1 = vertices[idx1];
		Vertex& v2 = vertices[idx2];

		// Вычисляем векторы ребер треугольника
		float edge1[3] = {
			v1.pos[0] - v0.pos[0],
			v1.pos[1] - v0.pos[1],
			v1.pos[2] - v0.pos[2]
		};
		float edge2[3] = {
			v2.pos[0] - v0.pos[0],
			v2.pos[1] - v0.pos[1],
			v2.pos[2] - v0.pos[2]
		};

		// Векторное произведение (edge1 x edge2) для получения нормали грани
		float normal[3] = {
			edge1[1] * edge2[2] - edge1[2] * edge2[1],
			edge1[2] * edge2[0] - edge1[0] * edge2[2],
			edge1[0] * edge2[1] - edge1[1] * edge2[0]
		};

		// Аккумулируем нормали к вершинам треугольника
		for (int j = 0; j < 3; ++j) {
			v0.normal[j] += normal[j];
			v1.normal[j] += normal[j];
			v2.normal[j] += normal[j];
		}
	}

	// Нормализуем полученные векторы нормалей для каждой вершины
	for (auto& v : vertices) {
		float length = std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2]);
		if (length > 0.0f) {
			v.normal[0] /= length;
			v.normal[1] /= length;
			v.normal[2] /= length;
		}
	}

	// ПОЛ
	const float floorY = -1.0f;
	const float floorS = 2.0f;
	std::vector<Vertex> floorVerts = {
		{{-floorS, floorY, -floorS}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
		{{ floorS, floorY, -floorS}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
		{{ floorS, floorY,  floorS}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		{{-floorS, floorY,  floorS}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
	};
	std::vector<uint16_t> floorIndices = {0, 2, 1,  3, 2, 0};
	s_cubeIndexCount   = (uint32_t)indices.size();
	s_floorIndexOffset = (uint32_t)indices.size();
	s_floorIndexCount  = (uint32_t)floorIndices.size();

	uint16_t floorBase = (uint16_t)vertices.size();
	for (auto& v : floorVerts) vertices.push_back(v);
	for (auto  i : floorIndices) indices.push_back(i + floorBase);

	// Создание staging буфера для геометрии
	VkDeviceSize vertexDataSize = sizeof(Vertex) * vertices.size();
	VkDeviceSize indexDataSize = sizeof(uint16_t) * indices.size();
	VkDeviceSize stagingSize = vertexDataSize + indexDataSize;

	VkBuffer stagingBuffer;
	VmaAllocation stagingAlloc;
	VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	stagingInfo.size = stagingSize;
	stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	VmaAllocationCreateInfo stagingAllocInfo{};
	stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

	VmaAllocationInfo stagingResult;
	if (vmaCreateBuffer(ctx.allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAlloc, &stagingResult) != VK_SUCCESS) {
		return false;
	}

	// Копируем данные в staging
	uint8_t* stagingPtr = (uint8_t*)stagingResult.pMappedData;
	std::memcpy(stagingPtr, vertices.data(), vertexDataSize);
	std::memcpy(stagingPtr + vertexDataSize, indices.data(), indexDataSize);

	// Создаем VRAM буферы
	VkBufferCreateInfo vbInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	vbInfo.size = vertexDataSize;
	vbInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // TRANSFER_DST

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; // Только GPU

	if (vmaCreateBuffer(ctx.allocator, &vbInfo, &vmaAllocInfo, &s_vertexBuffer, &s_vertexAlloc, nullptr) != VK_SUCCESS) {
		return false;
	}

	VkBufferCreateInfo ibInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	ibInfo.size = indexDataSize;
	ibInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // TRANSFER_DST

	if (vmaCreateBuffer(ctx.allocator, &ibInfo, &vmaAllocInfo, &s_indexBuffer, &s_indexAlloc, nullptr) != VK_SUCCESS) {
		return false;
	}

	// Одноразовый командный буфер для копирования
	VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	poolInfo.queueFamilyIndex = ctx.graphics_queue_index;
	VkCommandPool tempPool;
	vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &tempPool);

	VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cmdAllocInfo.commandPool = tempPool;
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandBufferCount = 1;
	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(ctx.device, &cmdAllocInfo, &cmd);

	VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &beginInfo);

	VkBufferCopy vertexCopy{};
	vertexCopy.srcOffset = 0;
	vertexCopy.dstOffset = 0;
	vertexCopy.size = vertexDataSize;
	vkCmdCopyBuffer(cmd, stagingBuffer, s_vertexBuffer, 1, &vertexCopy);

	VkBufferCopy indexCopy{};
	indexCopy.srcOffset = vertexDataSize;
	indexCopy.dstOffset = 0;
	indexCopy.size = indexDataSize;
	vkCmdCopyBuffer(cmd, stagingBuffer, s_indexBuffer, 1, &indexCopy);

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;
	VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence);
	vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, fence);
	vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

	vkDestroyFence(ctx.device, fence, nullptr);
	vkDestroyCommandPool(ctx.device, tempPool, nullptr);
	vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);

	s_indexCount = uint32_t(indices.size());

	// Динамический UBO
	// Получаем выравнивание для динамического UBO
	VkPhysicalDeviceProperties deviceProps{};
	vkGetPhysicalDeviceProperties(ctx.physical_device, &deviceProps);
	s_dynamicAlignment = deviceProps.limits.minUniformBufferOffsetAlignment;
	
	// Вычисляем размер одной структуры с учетом выравнивания
	s_modelUniformsSize = (sizeof(GlobalUniforms) + s_dynamicAlignment - 1) & ~(s_dynamicAlignment - 1);

	// Один буфер на 2 модели (куб и пол)
	const uint32_t maxModels = 2;
	VkDeviceSize bufferSize = s_modelUniformsSize * maxModels;

	VkBufferCreateInfo uboInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	uboInfo.size = bufferSize;
	uboInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

	VmaAllocationCreateInfo uboAllocInfo{};
	uboAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	uboAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

	VmaAllocationInfo resultAllocInfo{};
	if (vmaCreateBuffer(ctx.allocator, &uboInfo, &uboAllocInfo, &s_uniformBuffer, &s_uniformAlloc, &resultAllocInfo) != VK_SUCCESS) {
		return false;
	}
	s_uniformBufferMapped = resultAllocInfo.pMappedData;

	// Layout с UBO (binding 0) и текстурой (binding 1)
	VkDescriptorSetLayoutBinding bindings[2]{};

	// Динамический UBO
	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; // DYNAMIC
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	// Текстура
	bindings[1].binding = 1;
	bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo layoutCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	layoutCreateInfo.bindingCount = 2;
	layoutCreateInfo.pBindings = bindings;

	if (vkCreateDescriptorSetLayout(ctx.device, &layoutCreateInfo, nullptr, &s_descriptorSetLayout) != VK_SUCCESS) {
		return false;
	}

	// Пул дескрипторов с UBO и сэмплерами
	VkDescriptorPoolSize poolSizes[2]{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
	poolSizes[0].descriptorCount = 2;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = 2;

	VkDescriptorPoolCreateInfo descPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	descPoolInfo.poolSizeCount = 2;
	descPoolInfo.pPoolSizes = poolSizes;
	descPoolInfo.maxSets = 2;

	if (vkCreateDescriptorPool(ctx.device, &descPoolInfo, nullptr, &s_descriptorPool) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorSetAllocateInfo allocSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	allocSetInfo.descriptorPool = s_descriptorPool;
	allocSetInfo.descriptorSetCount = 2;
	VkDescriptorSetLayout layouts[2] = { s_descriptorSetLayout, s_descriptorSetLayout };
	allocSetInfo.pSetLayouts = layouts;

	VkDescriptorSet sets[2];
	if (vkAllocateDescriptorSets(ctx.device, &allocSetInfo, sets) != VK_SUCCESS) {
		return false;
	}
	s_descriptorSet      = sets[0];
	s_descriptorSetFloor = sets[1];

	// Записываем данные в дескрипторы
	// Информация о UBO (смещение 0, размер структуры)
	VkDescriptorBufferInfo uboInfoDesc{};
	uboInfoDesc.buffer = s_uniformBuffer;
	uboInfoDesc.offset = 0;
	uboInfoDesc.range = sizeof(GlobalUniforms); // ВАЖНО: sizeof структуры, а не выровненный размер

	// Информация о текстуре куба
	VkDescriptorImageInfo imageInfoCube{};
	imageInfoCube.sampler = s_textureCube.sampler;
	imageInfoCube.imageView = s_textureCube.view;
	imageInfoCube.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	// Информация о текстуре пола
	VkDescriptorImageInfo imageInfoFloor{};
	imageInfoFloor.sampler = s_textureFloor.sampler;
	imageInfoFloor.imageView = s_textureFloor.view;
	imageInfoFloor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkWriteDescriptorSet writes[4]{};

	// Набор для куба: UBO
	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = s_descriptorSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
	writes[0].descriptorCount = 1;
	writes[0].pBufferInfo = &uboInfoDesc;

	// Набор для куба: Текстура
	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = s_descriptorSet;
	writes[1].dstBinding = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[1].descriptorCount = 1;
	writes[1].pImageInfo = &imageInfoCube;

	// Набор для пола: UBO (тот же самый буфер, offset задается динамически)
	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = s_descriptorSetFloor;
	writes[2].dstBinding = 0;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
	writes[2].descriptorCount = 1;
	writes[2].pBufferInfo = &uboInfoDesc;

	// Набор для пола: Текстура
	writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[3].dstSet = s_descriptorSetFloor;
	writes[3].dstBinding = 1;
	writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[3].descriptorCount = 1;
	writes[3].pImageInfo = &imageInfoFloor;

	vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);

	// Создаем Pipeline Layout с поддержкой DescriptorSetLayout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &s_descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &s_pipelineLayout) != VK_SUCCESS) {
		return false;
	}

	// Graphics Pipeline
	VkPipelineShaderStageCreateInfo vertShaderStage{};
	vertShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStage.module = s_vertModule;
	vertShaderStage.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStage{};
	fragShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStage.module = s_fragModule;
	fragShaderStage.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStage, fragShaderStage };

	VkVertexInputBindingDescription bindingDesc{};
	bindingDesc.binding = 0;
	bindingDesc.stride = sizeof(Vertex);
	bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attrDesc[3];
	
	// Позиция (location = 0)
	attrDesc[0].location = 0;
	attrDesc[0].binding = 0;
	attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attrDesc[0].offset = offsetof(Vertex, pos);

	// Нормаль (location = 1)
	attrDesc[1].location = 1;
	attrDesc[1].binding = 0;
	attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attrDesc[1].offset = offsetof(Vertex, normal);

	// UV (location = 2)
	attrDesc[2].location = 2;
	attrDesc[2].binding = 0;
	attrDesc[2].format = VK_FORMAT_R32G32_SFLOAT; // ДВА ФЛОАТА
	attrDesc[2].offset = offsetof(Vertex, uv);

	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &bindingDesc;
	vertexInput.vertexAttributeDescriptionCount = 3;
	vertexInput.pVertexAttributeDescriptions = attrDesc;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo raster{};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.lineWidth = 1.0f;
	raster.cullMode = VK_CULL_MODE_BACK_BIT;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	VkPipelineMultisampleStateCreateInfo multisample{};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;

	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 2;
	dynamicState.pDynamicStates = dynamicStates;

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &raster;
	pipelineInfo.pMultisampleState = &multisample;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = s_pipelineLayout;
	pipelineInfo.renderPass = ctx.render_pass;
	pipelineInfo.subpass = 0;

	return vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &s_pipeline) == VK_SUCCESS;
}

void shutdown() {
	using namespace graphics::internal;
	auto& ctx = context;
	vkQueueWaitIdle(ctx.graphics_queue);

	auto destroyTex = [&](Texture& tex) {
        if (tex.sampler) vkDestroySampler(ctx.device, tex.sampler, nullptr);
        if (tex.view) vkDestroyImageView(ctx.device, tex.view, nullptr);
        if (tex.image) vmaDestroyImage(ctx.allocator, tex.image, tex.alloc);
    };
    destroyTex(s_textureCube);
    destroyTex(s_textureFloor);

	if (s_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx.device, s_descriptorPool, nullptr);
	if (s_descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, s_descriptorSetLayout, nullptr);
	if (s_uniformBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_uniformBuffer, s_uniformAlloc);

	if (s_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, s_pipeline, nullptr);
	if (s_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, s_pipelineLayout, nullptr);
	if (s_vertModule != VK_NULL_HANDLE) vkDestroyShaderModule(ctx.device, s_vertModule, nullptr);
	if (s_fragModule != VK_NULL_HANDLE) vkDestroyShaderModule(ctx.device, s_fragModule, nullptr);
	if (s_vertexBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_vertexBuffer, s_vertexAlloc);
	if (s_indexBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_indexBuffer, s_indexAlloc);
}

void update(double dt) {
    // Создаем свое окно
    ImGui::Begin("Laba #3");

    // Смена проекции
    ImGui::RadioButton("Perspective projection", (int*)&s_usePerspective, 1);
    ImGui::RadioButton("Ortographic projection", (int*)&s_usePerspective, 0);

    ImGui::Separator();

    // Смена поворота
    ImGui::Checkbox("Automatic rotation", &s_autoRotate);
    if (!s_autoRotate) {
        ImGui::SliderFloat3("Rotate (X, Y, Z)", s_rotation, -180.0f, 180.0f);
    }

	ImGui::Separator();
    ImGui::Text("Camera Settings");
    ImGui::SliderFloat3("Cam Pos", s_camPos, -3.0f, 3.0f);
    ImGui::SliderFloat3("Cam Rot (Pitch, Yaw, Roll)", s_camRot, -180.0f, 180.0f);
    ImGui::SliderFloat("FOV", &s_fov, 10.0f, 120.0f);

    ImGui::Separator();
    ImGui::Text("Lighting & Material");
    ImGui::SliderFloat3("Light Dir", s_lightDir, -1.0f, 1.0f);
    ImGui::ColorEdit3("Light Color", s_lightColor);
    ImGui::ColorEdit3("Ambient Light", s_ambientLight);
    ImGui::ColorEdit3("Specular Color (Ks)", s_specularColor);
    ImGui::SliderFloat("Shininess", &s_shininess, 1.0f, 256.0f);

    ImGui::End();

	if (s_autoRotate) {
        s_rotation[0] = (float)dt * 50.0f;
        s_rotation[1] = (float)dt * 35.0f;
    }
}

void render(const graphics::internal::FrameData& fd) {
	using namespace graphics::internal;
	auto& ctx = context;

	if (ctx.swapchain_extent.width == 0 || ctx.swapchain_extent.height == 0) return;

	// Переводим углы из градусов в радианы
	float radX = s_rotation[0] * 3.14159265f / 180.0f;
	float radY = s_rotation[1] * 3.14159265f / 180.0f;
	float radZ = s_rotation[2] * 3.14159265f / 180.0f;

	float cX = std::cos(radX), sX = std::sin(radX);
	float cY = std::cos(radY), sY = std::sin(radY);
	float cZ = std::cos(radZ), sZ = std::sin(radZ);

	// Матрица поворота Model (комбинация вращений Z * Y * X)
	std::array<float, 16> model = {
		cZ * cY,                  sZ * cY,                  -sY,     0.0f,
		cZ * sY * sX - sZ * cX,   sZ * sY * sX + cZ * cX,   cY * sX, 0.0f,
		cZ * sY * cX + sZ * sX,   sZ * sY * cX - cZ * sX,   cY * cX, 0.0f,
		0.0f,                     0.0f,                     0.0f,    1.0f
	};

	float camPitch = s_camRot[0] * 3.14159265f / 180.0f;
	float camYaw   = s_camRot[1] * 3.14159265f / 180.0f;
	float camRoll  = s_camRot[2] * 3.14159265f / 180.0f;
	
	// Формирование матриц поворота камеры вокруг осей X и Y
	float cp = std::cos(camPitch), sp = std::sin(camPitch);
	float cy = std::cos(camYaw),   sy = std::sin(camYaw);
	float cr = std::cos(camRoll),  sr = std::sin(camRoll);

	// Матрица View (комбинация вращения и перемещения камеры)
	std::array<float, 16> view = {
		cy * cr + sy * sp * sr,   cp * sr,   -sy * cr + cy * sp * sr,   0.0f,
		-cy * sr + sy * sp * cr,  cp * cr,   sy * sr + cy * sp * cr,    0.0f,
		sy * cp,                 -sp,        cy * cp,                   0.0f,
		0.0f,                     0.0f,      0.0f,                      1.0f
	};

	float tx = -s_camPos[0];
	float ty = -s_camPos[1];
	float tz = -s_camPos[2];

	view[12] = tx * view[0] + ty * view[4] + tz * view[8];
	view[13] = tx * view[1] + ty * view[5] + tz * view[9];
	view[14] = tx * view[2] + ty * view[6] + tz * view[10];
	view[15] = 1.0f;

	// Матрица Projection
	float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);
	std::array<float, 16> proj;

	if (s_usePerspective) {
		// Перспективная проекция
		float fovy = s_fov * 3.14159265f / 180.0f;
		float f = 1.0f / std::tan(fovy * 0.5f);
		float zNear = 0.1f, zFar = 100.0f;

		proj = {
			f / aspect, 0.0f, 0.0f, 0.0f,
			0.0f, -f, 0.0f, 0.0f, // -f инвертирует Y для Vulkan
			0.0f, 0.0f, zFar / (zNear - zFar), -1.0f,
			0.0f, 0.0f, (zNear * zFar) / (zNear - zFar), 0.0f
		};
	} else {
		// Ортографическая проекция
		float size = 2.0f;
		float left = -size * aspect, right = size * aspect;
		float bottom = size, top = -size; // top отрицательный для Vulkan Y
		float zNear = 0.1f, zFar = 100.0f;

		proj = {
			2.0f / (right - left), 0.0f, 0.0f, 0.0f,
			0.0f, 2.0f / (top - bottom), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f / (zNear - zFar), 0.0f,
			-(right + left)/(right - left), -(top + bottom)/(top - bottom), zNear / (zNear - zFar), 1.0f
		};
	}

	// Перемножение матриц: MVP = Proj * View * Model
	auto multiplyMatrices = [](const std::array<float,16>& A, const std::array<float,16>& B) {
		std::array<float,16> C{};
		for (int col = 0; col < 4; ++col) {
			for (int row = 0; row < 4; ++row) {
				float sum = 0.0f;
				for (int k = 0; k < 4; ++k) {
					sum += A[k * 4 + row] * B[col * 4 + k];
				}
				C[col * 4 + row] = sum;
			}
		}
		return C;
	};

	auto viewMode = multiplyMatrices(view, model);
	auto mvp = multiplyMatrices(proj, viewMode);

	// Обновляем динамический UBO
	GlobalUniforms uboCube{};
	std::memcpy(uboCube.mvp, mvp.data(), sizeof(mvp));
	std::memcpy(uboCube.model, model.data(), sizeof(model));
	std::memcpy(uboCube.ambientLight, s_ambientLight, sizeof(s_ambientLight));
	std::memcpy(uboCube.lightColor, s_lightColor, sizeof(s_lightColor));
	std::memcpy(uboCube.lightDir, s_lightDir, sizeof(s_lightDir));
	std::memcpy(uboCube.cameraPos, s_camPos, sizeof(s_camPos));
	std::memcpy(uboCube.specularColor, s_specularColor, sizeof(s_specularColor));
	uboCube.shininess = s_shininess;
	uboCube.pad1 = 0.0f; uboCube.pad2 = 0.0f; uboCube.pad3 = 0.0f;

	// Пол не вращается: model = identity
	std::array<float,16> floorModel = {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};
	auto floorMvp = multiplyMatrices(proj, multiplyMatrices(view, floorModel));

	GlobalUniforms uboFloor = uboCube; // копируем освещение/камеру
	std::memcpy(uboFloor.mvp,   floorMvp.data(),  sizeof(floorMvp));
	std::memcpy(uboFloor.model, floorModel.data(), sizeof(floorModel));

	// Копируем данные в один буфер с учетом выравнивания
	uint8_t* uboPtr = (uint8_t*)s_uniformBufferMapped;
	std::memcpy(uboPtr, &uboCube, sizeof(GlobalUniforms));
	std::memcpy(uboPtr + s_modelUniformsSize, &uboFloor, sizeof(GlobalUniforms));

	// Запись буфера команд
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

	const VkClearValue clearValues[2] = {
		{ .color = { {0.15f, 0.15f, 0.15f, 1.0f} } },
		{ .depthStencil = {1.0f, 0} }
	};

	VkRenderPassBeginInfo renderPassBegin{};
	renderPassBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBegin.renderPass = ctx.render_pass;
	renderPassBegin.framebuffer = fd.framebuffer;
	renderPassBegin.renderArea.offset = {0,0};
	renderPassBegin.renderArea.extent = ctx.swapchain_extent;
	renderPassBegin.clearValueCount = 2;
	renderPassBegin.pClearValues = clearValues;

	vkCmdBeginRenderPass(fd.command_buffer, &renderPassBegin, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = float(ctx.swapchain_extent.width);
	viewport.height = float(ctx.swapchain_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = {0,0};
	scissor.extent = ctx.swapchain_extent;
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pipeline);

	VkDeviceSize offsets[] = {0};
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &s_vertexBuffer, offsets);
	vkCmdBindIndexBuffer(fd.command_buffer, s_indexBuffer, 0, VK_INDEX_TYPE_UINT16);

	// Привязываем дескрипторы с динамическим сдвигом
	// Куб (сдвиг = 0)
	uint32_t dynamicOffsetCube = 0;
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        s_pipelineLayout, 0, 1, &s_descriptorSet, 
	                        1, &dynamicOffsetCube); // Передаем сдвиг
	vkCmdDrawIndexed(fd.command_buffer, s_cubeIndexCount, 1, 0, 0, 0);

	// Пол (сдвиг = s_modelUniformsSize)
	uint32_t dynamicOffsetFloor = static_cast<uint32_t>(s_modelUniformsSize);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        s_pipelineLayout, 0, 1, &s_descriptorSetFloor, 
	                        1, &dynamicOffsetFloor); // Передаем сдвиг!
	vkCmdDrawIndexed(fd.command_buffer, s_floorIndexCount, 1, s_floorIndexOffset, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application