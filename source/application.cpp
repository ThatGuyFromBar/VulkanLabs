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
};
static VkBuffer s_uniformBuffer = VK_NULL_HANDLE;
static VmaAllocation s_uniformAlloc = {};
static void* s_uniformBufferMapped = nullptr; // Отображенная память UBO

static VkDescriptorSetLayout s_descriptorSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool s_descriptorPool = VK_NULL_HANDLE;
static VkDescriptorSet s_descriptorSet = VK_NULL_HANDLE;

static VkPipeline s_pipeline = VK_NULL_HANDLE; // конвейер
static VkPipelineLayout s_pipelineLayout = VK_NULL_HANDLE; // макет конвейера
static VkShaderModule s_vertModule = VK_NULL_HANDLE; // вершинный шейдер
static VkShaderModule s_fragModule = VK_NULL_HANDLE; // фрагментный шейдер

// UI пользователя
static bool s_usePerspective = true; // смена вида
static float s_rotation[3] = {0.0f, 0.0f, 0.0f}; // поворот куба
static bool s_autoRotate = false; // автовращение

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

	// ЗАДАЕМ КУБ
	struct Vertex { float pos[3]; float color[3]; }; // позиция + цвет

	const std::vector<Vertex> vertices = {
		// Передние вершины
		{{-0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}},
		{{ 0.5f, -0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
		{{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
		{{-0.5f,  0.5f,  0.5f}, {1.0f, 1.0f, 0.0f}},
		// Задние вершины
		{{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},
		{{ 0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}},
		{{ 0.5f,  0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}},
		{{-0.5f,  0.5f, -0.5f}, {0.2f, 0.2f, 0.2f}},
	};

	// Грани (по 2 треугольника)
	const std::vector<uint16_t> indices = {
		0, 1, 2,  2, 3, 0, // передняя грань
		1, 5, 6,  6, 2, 1, // правая
		5, 4, 7,  7, 6, 5, // задняя
		4, 0, 3,  3, 7, 4, // левая
		3, 2, 6,  6, 7, 3, // верхняя
		4, 5, 1,  1, 0, 4  // нижняя
	};

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

	// Создаем Descriptor Set Layout
	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

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
	poolSize.descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = 1;

	if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &s_descriptorPool) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorSetAllocateInfo allocSetInfo{};
	allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocSetInfo.descriptorPool = s_descriptorPool;
	allocSetInfo.descriptorSetCount = 1;
	allocSetInfo.pSetLayouts = &s_descriptorSetLayout;

	if (vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &s_descriptorSet) != VK_SUCCESS) {
		return false;
	}

	// Связываем Uniform Buffer с Descriptor Set
	VkDescriptorBufferInfo bufferInfo{};
	bufferInfo.buffer = s_uniformBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = sizeof(GlobalUniforms);

	VkWriteDescriptorSet descriptorWrite{};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = s_descriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &bufferInfo;

	vkUpdateDescriptorSets(ctx.device, 1, &descriptorWrite, 0, nullptr);

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

	VkVertexInputAttributeDescription attrDesc[2];
	attrDesc[0].location = 0;
	attrDesc[0].binding = 0;
	attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attrDesc[0].offset = offsetof(Vertex, pos);

	attrDesc[1].location = 1;
	attrDesc[1].binding = 0;
	attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attrDesc[1].offset = offsetof(Vertex, color);

	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &bindingDesc;
	vertexInput.vertexAttributeDescriptionCount = 2;
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

	if (s_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, s_pipeline, nullptr);
	if (s_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, s_pipelineLayout, nullptr);
	if (s_vertModule != VK_NULL_HANDLE) vkDestroyShaderModule(ctx.device, s_vertModule, nullptr);
	if (s_fragModule != VK_NULL_HANDLE) vkDestroyShaderModule(ctx.device, s_fragModule, nullptr);
	if (s_vertexBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_vertexBuffer, s_vertexAlloc);
	if (s_indexBuffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, s_indexBuffer, s_indexAlloc);
}

void update(double dt) {
    // Создаем свое окно
    ImGui::Begin("Laba #1");

    // Смена проекции
    ImGui::RadioButton("Perspective projection", (int*)&s_usePerspective, 1);
    ImGui::RadioButton("Ortographic projection", (int*)&s_usePerspective, 0);

    ImGui::Separator();

    // Смена поворота
    ImGui::Checkbox("Automatic rotation", &s_autoRotate);
    if (!s_autoRotate) {
        ImGui::SliderFloat3("Rotate (X, Y, Z)", s_rotation, -180.0f, 180.0f);
    }

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

	// Матрица View (камера)
	std::array<float, 16> view = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, -3.5f, 1.0f
	};

	// Матрица Projection
	float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);
	std::array<float, 16> proj;

	if (s_usePerspective) {
		// Перспективная проекция
		float fovy = 45.0f * 3.14159265f / 180.0f;
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
	std::memcpy(s_uniformBufferMapped, &ubo, sizeof(ubo));

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
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pipelineLayout, 0, 1, &s_descriptorSet, 0, nullptr);

	vkCmdDrawIndexed(fd.command_buffer, s_indexCount, 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application