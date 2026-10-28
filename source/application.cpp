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

namespace application {

// Глобальные переменные
// Буфер вершин
static VkBuffer s_vertexBuffer = VK_NULL_HANDLE;
static VmaAllocation s_vertexAlloc = {};
// Буфер индексов
static VkBuffer s_indexBuffer = VK_NULL_HANDLE;
static VmaAllocation s_indexAlloc = {};
static uint32_t s_indexCount = 0;

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

bool initialize() {
	using namespace graphics::internal;
	auto& ctx = context;

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
		float color[3];  // Цвет / альбедо
	};

	std::vector<Vertex> vertices = {
		// 1. Передняя грань
		{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
		{{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
		{{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
		{{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},

		// 2. Задняя грань
		{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
		{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
		{{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
		{{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},

		// 3. Левая грань
		{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
		{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
		{{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
		{{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},

		// 4. Правая грань
		{{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}},
		{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}},
		{{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}},
		{{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}},

		// 5. Нижняя грань
		{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}},
		{{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}},
		{{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}},
		{{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}},

		// 6. Верхняя грань
		{{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f}},
		{{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f}},
		{{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f}},
		{{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f}},
	};

	// Индексы для 6 граней (36 индексов)[cite: 1]
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
		{{-floorS, floorY, -floorS}, {0.0f, 1.0f, 0.0f}, {0.35f, 0.35f, 0.35f}},
		{{ floorS, floorY, -floorS}, {0.0f, 1.0f, 0.0f}, {0.35f, 0.35f, 0.35f}},
		{{ floorS, floorY,  floorS}, {0.0f, 1.0f, 0.0f}, {0.35f, 0.35f, 0.35f}},
		{{-floorS, floorY,  floorS}, {0.0f, 1.0f, 0.0f}, {0.35f, 0.35f, 0.35f}},
	};
	std::vector<uint16_t> floorIndices = {0, 2, 1,  3, 2, 0};
	s_cubeIndexCount   = (uint32_t)indices.size();
	s_floorIndexOffset = (uint32_t)indices.size();
	s_floorIndexCount  = (uint32_t)floorIndices.size();

	uint16_t floorBase = (uint16_t)vertices.size();
	for (auto& v : floorVerts) vertices.push_back(v);
	for (auto  i : floorIndices) indices.push_back(i + floorBase);

	// Создаем буфер вершин
	VkBufferCreateInfo vbInfo{};
	vbInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vbInfo.size = sizeof(Vertex) * vertices.size();
	vbInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	if (vmaCreateBuffer(ctx.allocator, &vbInfo, &vmaAllocInfo, &s_vertexBuffer, &s_vertexAlloc, nullptr) != VK_SUCCESS) {
		return false;
	}

	void* mapped = nullptr;
	vmaMapMemory(ctx.allocator, s_vertexAlloc, &mapped);
	std::memcpy(mapped, vertices.data(), vbInfo.size);
	vmaUnmapMemory(ctx.allocator, s_vertexAlloc);

	// Аналогично буфер индексов
	VkBufferCreateInfo ibInfo{};
	ibInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	ibInfo.size = sizeof(uint16_t) * indices.size();
	ibInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

	if (vmaCreateBuffer(ctx.allocator, &ibInfo, &vmaAllocInfo, &s_indexBuffer, &s_indexAlloc, nullptr) != VK_SUCCESS) {
		return false;
	}

	vmaMapMemory(ctx.allocator, s_indexAlloc, &mapped);
	std::memcpy(mapped, indices.data(), ibInfo.size);
	vmaUnmapMemory(ctx.allocator, s_indexAlloc);

	s_indexCount = uint32_t(indices.size());

	// Создаем Uniform Buffer для матрицы MVP
	VkBufferCreateInfo uboInfo{};
	uboInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	uboInfo.size = sizeof(GlobalUniforms);
	uboInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

	VmaAllocationCreateInfo uboAllocInfo{};
	uboAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	uboAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

	VmaAllocationInfo resultAllocInfo{};
	if (vmaCreateBuffer(ctx.allocator, &uboInfo, &uboAllocInfo, &s_uniformBuffer, &s_uniformAlloc, &resultAllocInfo) != VK_SUCCESS) {
		return false;
	}
	s_uniformBufferMapped = resultAllocInfo.pMappedData;
	// ПОЛ
	VmaAllocationInfo resultAllocInfoFloor{};
	if (vmaCreateBuffer(ctx.allocator, &uboInfo, &uboAllocInfo, &s_uniformBufferFloor, &s_uniformAllocFloor, &resultAllocInfoFloor) != VK_SUCCESS) {
		return false;
	}
	s_uniformBufferMappedFloor = resultAllocInfoFloor.pMappedData;

	// Создаем Descriptor Set Layout
	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
	layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutCreateInfo.bindingCount = 1;
	layoutCreateInfo.pBindings = &uboLayoutBinding;

	if (vkCreateDescriptorSetLayout(ctx.device, &layoutCreateInfo, nullptr, &s_descriptorSetLayout) != VK_SUCCESS) {
		return false;
	}

	// Создаем Descriptor Pool и выделяем Descriptor Set
	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSize.descriptorCount = 2;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = 2;

	if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &s_descriptorPool) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorSetAllocateInfo allocSetInfo{};
	allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
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

	// Связываем Uniform Buffer с Descriptor Set
		VkDescriptorBufferInfo bufferInfo{};
	bufferInfo.buffer = s_uniformBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = sizeof(GlobalUniforms);

	VkDescriptorBufferInfo bufferInfoFloor{};
	bufferInfoFloor.buffer = s_uniformBufferFloor;
	bufferInfoFloor.offset = 0;
	bufferInfoFloor.range = sizeof(GlobalUniforms);

	VkWriteDescriptorSet writes[2]{};
	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = s_descriptorSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[0].descriptorCount = 1;
	writes[0].pBufferInfo = &bufferInfo;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = s_descriptorSetFloor;
	writes[1].dstBinding = 0;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[1].descriptorCount = 1;
	writes[1].pBufferInfo = &bufferInfoFloor;

	vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

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

	VkVertexInputAttributeDescription attrDesc[3]; // Теперь у нас 3 атрибута!
	
	// 1. Позиция (location = 0)
	attrDesc[0].location = 0;
	attrDesc[0].binding = 0;
	attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attrDesc[0].offset = offsetof(Vertex, pos);

	// 2. Нормаль (location = 1)
	attrDesc[1].location = 1;
	attrDesc[1].binding = 0;
	attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attrDesc[1].offset = offsetof(Vertex, normal);

	// 3. Цвет / Альбедо (location = 2)
	attrDesc[2].location = 2;
	attrDesc[2].binding = 0;
	attrDesc[2].format = VK_FORMAT_R32G32B32_SFLOAT;
	attrDesc[2].offset = offsetof(Vertex, color);

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

	if (s_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx.device, s_descriptorPool, nullptr);
	if (s_descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, s_descriptorSetLayout, nullptr);
	if (s_uniformBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_uniformBuffer, s_uniformAlloc);
	if (s_uniformBufferFloor != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_uniformBufferFloor, s_uniformAllocFloor);

	if (s_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, s_pipeline, nullptr);
	if (s_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, s_pipelineLayout, nullptr);
	if (s_vertModule != VK_NULL_HANDLE) vkDestroyShaderModule(ctx.device, s_vertModule, nullptr);
	if (s_fragModule != VK_NULL_HANDLE) vkDestroyShaderModule(ctx.device, s_fragModule, nullptr);
	if (s_vertexBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_vertexBuffer, s_vertexAlloc);
	if (s_indexBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_indexBuffer, s_indexAlloc);
}

void update(double dt) {
    // Создаем свое окно
    ImGui::Begin("Laba #2");

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

	// Обновляем буфер перед записью команд
	GlobalUniforms ubo{};
	std::memcpy(ubo.mvp, mvp.data(), sizeof(mvp));
	std::memcpy(ubo.model, model.data(), sizeof(model));
    
	std::memcpy(ubo.ambientLight, s_ambientLight, sizeof(s_ambientLight));
	std::memcpy(ubo.lightColor, s_lightColor, sizeof(s_lightColor));
	std::memcpy(ubo.lightDir, s_lightDir, sizeof(s_lightDir));
	std::memcpy(ubo.cameraPos, s_camPos, sizeof(s_camPos));
	std::memcpy(ubo.specularColor, s_specularColor, sizeof(s_specularColor));
	ubo.shininess = s_shininess;
	ubo.pad1 = 0.0f; ubo.pad2 = 0.0f; ubo.pad3 = 0.0f;

	std::memcpy(s_uniformBufferMapped, &ubo, sizeof(ubo));
	// === НОВОЕ: UBO для пола ===
	// Пол не вращается: model = identity
	std::array<float,16> floorModel = {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};
	auto floorMvp = multiplyMatrices(proj, multiplyMatrices(view, floorModel));

	GlobalUniforms uboFloor = ubo; // копируем освещение/камеру
	std::memcpy(uboFloor.mvp,   floorMvp.data(),  sizeof(floorMvp));
	std::memcpy(uboFloor.model, floorModel.data(), sizeof(floorModel));
	std::memcpy(s_uniformBufferMappedFloor, &uboFloor, sizeof(uboFloor));

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

	// Привязываем Descriptor Set с обновленными матрицами
	// Куб
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        s_pipelineLayout, 0, 1, &s_descriptorSet, 0, nullptr);
	vkCmdDrawIndexed(fd.command_buffer, s_cubeIndexCount, 1, 0, 0, 0);

	// Пол
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        s_pipelineLayout, 0, 1, &s_descriptorSetFloor, 0, nullptr);
	vkCmdDrawIndexed(fd.command_buffer, s_floorIndexCount, 1, s_floorIndexOffset, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application