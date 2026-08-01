#include "vulkan_context.hpp"
#include "vulkan_buffer.hpp"
#include "vulkan_matmul.hpp"
#include <vulkan/vulkan.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>

using namespace stakml::vulkan;

static std::vector<uint32_t> loadShader(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open shader: " + path);
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

// ─────────────────────────────────────────────────────────────────────────
// Test 1: vecadd
// ─────────────────────────────────────────────────────────────────────────
static bool runVecAddTest(VulkanContext& ctx) {
    const uint32_t N = 1024;
    VkDeviceSize bufSize = N * sizeof(float);
    VkDevice device = ctx.device();

    std::vector<float> a(N), b(N), expected(N);
    for (uint32_t i = 0; i < N; i++) {
        a[i] = float(i);
        b[i] = float(i) * 2.0f;
        expected[i] = a[i] + b[i];
    }

    VulkanBuffer bufA(device, ctx.physicalDevice(), bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer bufB(device, ctx.physicalDevice(), bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VulkanBuffer bufC(device, ctx.physicalDevice(), bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    void* mapped = bufA.map(); memcpy(mapped, a.data(), bufSize); bufA.unmap();
    mapped = bufB.map(); memcpy(mapped, b.data(), bufSize); bufB.unmap();

    auto code = loadShader("shaders/vecadd.spv");
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = code.size() * sizeof(uint32_t);
    shaderInfo.pCode = code.data();
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");

    VkDescriptorSetLayoutBinding bindings[3]{};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    VkDescriptorSetLayout descSetLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descSetLayout);

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &descSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushRange;
    VkPipelineLayout pipeLayout;
    vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &pipeLayout);

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage = stageInfo;
    pipeInfo.layout = pipeLayout;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create compute pipeline");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);

    VkDescriptorSetAllocateInfo dsAllocInfo{};
    dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAllocInfo.descriptorPool = descPool;
    dsAllocInfo.descriptorSetCount = 1;
    dsAllocInfo.pSetLayouts = &descSetLayout;
    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet);

    VkBuffer buffers[3] = { bufA.handle(), bufB.handle(), bufC.handle() };
    VkWriteDescriptorSet writes[3]{};
    VkDescriptorBufferInfo bufInfos[3]{};
    for (int i = 0; i < 3; i++) {
        bufInfos[i].buffer = buffers[i];
        bufInfos[i].offset = 0;
        bufInfos[i].range = bufSize;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = ctx.computeQueueFamily();
    VkCommandPool cmdPool;
    vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmdBuf, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &N);
    vkCmdDispatch(cmdBuf, (N + 255) / 256, 1, 1);

    vkEndCommandBuffer(cmdBuf);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    vkQueueSubmit(ctx.computeQueue(), 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    mapped = bufC.map();
    float* result = reinterpret_cast<float*>(mapped);
    bool ok = true;
    for (uint32_t i = 0; i < N; i++) {
        if (std::fabs(result[i] - expected[i]) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << result[i]
                      << ", expected " << expected[i] << "\n";
            ok = false;
            break;
        }
    }
    bufC.unmap();

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    std::cout << "vecadd on GPU (780M): " << (ok ? "PASSED" : "FAILED") << ", N=" << N << "\n";
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────
// Test 2: matmul — now exercises the TILED shader. Non-square, non-uniform
// values, AND dimensions that are NOT multiples of 16 (the tile size) —
// this deliberately forces the tile-boundary padding logic (the "pad with
// 0 when the tile overhangs" branches in matmul.comp) to actually run.
// ─────────────────────────────────────────────────────────────────────────
static bool runMatmulTest(VulkanContext& ctx) {
    const size_t M = 20, K = 35, N = 18;   // none are multiples of 16 — exercises tile edges

    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), expected(M * N, 0.0f);

    for (size_t i = 0; i < M * K; i++) A[i] = float((i % 13) + 1) * 0.1f;
    for (size_t i = 0; i < K * N; i++) B[i] = float((i % 7) + 1) * 0.2f;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    VulkanMatmul matmul(ctx);
    matmul.run(A.data(), B.data(), C.data(), M, K, N);

    bool ok = true;
    for (size_t i = 0; i < M * N; i++) {
        if (std::fabs(C[i] - expected[i]) > 1e-2f) {
            std::cerr << "Mismatch at flat index " << i
                       << " (row " << i / N << ", col " << i % N << "): got "
                       << C[i] << ", expected " << expected[i] << "\n";
            ok = false;
        }
    }

    std::cout << "matmul on GPU (780M): " << (ok ? "PASSED" : "FAILED")
               << ", " << M << "x" << K << " @ " << K << "x" << N
               << " (tiled, non-multiple-of-16 dims)\n";
    return ok;
}

int main() {
    try {
        VulkanContext ctx;

        bool vecaddOk = runVecAddTest(ctx);
        bool matmulOk = runMatmulTest(ctx);

        return (vecaddOk && matmulOk) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}