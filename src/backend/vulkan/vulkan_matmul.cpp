#include "vulkan_matmul.hpp"
#include "vulkan_buffer.hpp"
#include <array>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace stakml::vulkan {

static std::vector<uint32_t> loadShader(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open shader: " + path);
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

VulkanMatmul::VulkanMatmul(VulkanContext& ctx) : ctx_(ctx) {
    auto code = loadShader("shaders/matmul.spv");
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = code.size() * sizeof(uint32_t);
    shaderInfo.pCode = code.data();
    if (vkCreateShaderModule(ctx_.device(), &shaderInfo, nullptr, &shaderModule_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");

    createDescriptorSetLayout();
    createPipeline();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx_.computeQueueFamily();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx_.device(), &poolInfo, nullptr, &commandPool_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo dpoolInfo{};
    dpoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpoolInfo.poolSizeCount = 1;
    dpoolInfo.pPoolSizes = &poolSize;
    dpoolInfo.maxSets = 1;
    dpoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(ctx_.device(), &dpoolInfo, nullptr, &descriptorPool_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool");

    VkDescriptorSetAllocateInfo dsAllocInfo{};
    dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAllocInfo.descriptorPool = descriptorPool_;
    dsAllocInfo.descriptorSetCount = 1;
    dsAllocInfo.pSetLayouts = &dsLayout_;
    if (vkAllocateDescriptorSets(ctx_.device(), &dsAllocInfo, &descriptorSet_) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate descriptor set");

    createCommandBuffer();
}

VulkanMatmul::~VulkanMatmul() {
    if (commandPool_) vkDestroyCommandPool(ctx_.device(), commandPool_, nullptr);
    if (descriptorPool_) vkDestroyDescriptorPool(ctx_.device(), descriptorPool_, nullptr);
    if (batchDescriptorPool_) vkDestroyDescriptorPool(ctx_.device(), batchDescriptorPool_, nullptr);
    if (pipeline_) vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(ctx_.device(), pipelineLayout_, nullptr);
    if (dsLayout_) vkDestroyDescriptorSetLayout(ctx_.device(), dsLayout_, nullptr);
    if (shaderModule_) vkDestroyShaderModule(ctx_.device(), shaderModule_, nullptr);
}

void VulkanMatmul::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(ctx_.device(), &layoutInfo, nullptr, &dsLayout_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor set layout");
}

void VulkanMatmul::createPipeline() {
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule_;
    stageInfo.pName = "main";

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 3 * sizeof(uint32_t);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &dsLayout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(ctx_.device(), &layoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout");

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout_;

    if (vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create compute pipeline");
}

void VulkanMatmul::createCommandBuffer() {
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool_;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx_.device(), &cmdAllocInfo, &commandBuffer_) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffer");
}

// ── Single-call path (unchanged behavior from before) ─────────────────────

void VulkanMatmul::ensureBuffers(VkDeviceSize sizeA, VkDeviceSize sizeB, VkDeviceSize sizeC) {
    bool grew = false;
    if (sizeA > capacityA_) {
        bufA_ = std::make_unique<VulkanBuffer>(ctx_.device(), ctx_.physicalDevice(), sizeA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        capacityA_ = sizeA; grew = true;
    }
    if (sizeB > capacityB_) {
        bufB_ = std::make_unique<VulkanBuffer>(ctx_.device(), ctx_.physicalDevice(), sizeB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        capacityB_ = sizeB; grew = true;
    }
    if (sizeC > capacityC_) {
        bufC_ = std::make_unique<VulkanBuffer>(ctx_.device(), ctx_.physicalDevice(), sizeC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        capacityC_ = sizeC; grew = true;
    }
    if (grew) updateDescriptorSet();
}

void VulkanMatmul::updateDescriptorSet() {
    std::array<VkDescriptorBufferInfo, 3> bufferInfos{};
    bufferInfos[0] = { bufA_->handle(), 0, VK_WHOLE_SIZE };
    bufferInfos[1] = { bufB_->handle(), 0, VK_WHOLE_SIZE };
    bufferInfos[2] = { bufC_->handle(), 0, VK_WHOLE_SIZE };
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void VulkanMatmul::run(const float* A, const float* B, float* C,
                        size_t M, size_t K, size_t N) {
    VkDeviceSize sizeA = M * K * sizeof(float);
    VkDeviceSize sizeB = K * N * sizeof(float);
    VkDeviceSize sizeC = M * N * sizeof(float);
    ensureBuffers(sizeA, sizeB, sizeC);

    void* mapped = bufA_->map(); std::memcpy(mapped, A, sizeA); bufA_->unmap();
    mapped = bufB_->map(); std::memcpy(mapped, B, sizeB); bufB_->unmap();

    vkResetCommandBuffer(commandBuffer_, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &beginInfo);

    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                             0, 1, &descriptorSet_, 0, nullptr);
    uint32_t pushConsts[3] = { (uint32_t)M, (uint32_t)K, (uint32_t)N };
    vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConsts), pushConsts);
    uint32_t groupsX = ((uint32_t)N + 15) / 16;
    uint32_t groupsY = ((uint32_t)M + 15) / 16;
    vkCmdDispatch(commandBuffer_, groupsX, groupsY, 1);
    vkEndCommandBuffer(commandBuffer_);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;
    vkQueueSubmit(ctx_.computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx_.computeQueue());

    mapped = bufC_->map(); std::memcpy(C, mapped, sizeC); bufC_->unmap();
}

// ── Batch path — the actual experiment ─────────────────────────────────────

void VulkanMatmul::ensureBatchCapacity(size_t count) {
    // Descriptor pool: Vulkan pools can't grow in place, so if we need
    // more sets than the pool currently supports, destroy and recreate
    // it sized for `count`. This only happens when a batch is larger
    // than any previous one.
    if (count > batchPoolCapacity_) {
        if (batchDescriptorPool_) vkDestroyDescriptorPool(ctx_.device(), batchDescriptorPool_, nullptr);

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(count * 3);

        VkDescriptorPoolCreateInfo dpoolInfo{};
        dpoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpoolInfo.poolSizeCount = 1;
        dpoolInfo.pPoolSizes = &poolSize;
        dpoolInfo.maxSets = static_cast<uint32_t>(count);
        if (vkCreateDescriptorPool(ctx_.device(), &dpoolInfo, nullptr, &batchDescriptorPool_) != VK_SUCCESS)
            throw std::runtime_error("Failed to create batch descriptor pool");

        std::vector<VkDescriptorSetLayout> layouts(count, dsLayout_);
        batchSets_.assign(count, VK_NULL_HANDLE);

        VkDescriptorSetAllocateInfo dsAllocInfo{};
        dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAllocInfo.descriptorPool = batchDescriptorPool_;
        dsAllocInfo.descriptorSetCount = static_cast<uint32_t>(count);
        dsAllocInfo.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(ctx_.device(), &dsAllocInfo, batchSets_.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate batch descriptor sets");

        batchPoolCapacity_ = count;
    }

    if (batchBufA_.size() < count) {
        batchBufA_.resize(count);
        batchBufB_.resize(count);
        batchBufC_.resize(count);
        batchCapA_.resize(count, 0);
        batchCapB_.resize(count, 0);
        batchCapC_.resize(count, 0);
    }
}

void VulkanMatmul::runBatch(const std::vector<MatmulJob>& jobs) {
    if (jobs.empty()) return;
    ensureBatchCapacity(jobs.size());

    // Upload phase — grow per-slot buffers if needed, upload A/B, rebind
    // that slot's descriptor set to the (possibly new) buffer handles.
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto& job = jobs[i];
        VkDeviceSize sizeA = job.M * job.K * sizeof(float);
        VkDeviceSize sizeB = job.K * job.N * sizeof(float);
        VkDeviceSize sizeC = job.M * job.N * sizeof(float);

        bool grew = false;
        if (sizeA > batchCapA_[i]) {
            batchBufA_[i] = std::make_unique<VulkanBuffer>(ctx_.device(), ctx_.physicalDevice(), sizeA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            batchCapA_[i] = sizeA; grew = true;
        }
        if (sizeB > batchCapB_[i]) {
            batchBufB_[i] = std::make_unique<VulkanBuffer>(ctx_.device(), ctx_.physicalDevice(), sizeB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            batchCapB_[i] = sizeB; grew = true;
        }
        if (sizeC > batchCapC_[i]) {
            batchBufC_[i] = std::make_unique<VulkanBuffer>(ctx_.device(), ctx_.physicalDevice(), sizeC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            batchCapC_[i] = sizeC; grew = true;
        }

        if (grew) {
            std::array<VkDescriptorBufferInfo, 3> bufferInfos{};
            bufferInfos[0] = { batchBufA_[i]->handle(), 0, VK_WHOLE_SIZE };
            bufferInfos[1] = { batchBufB_[i]->handle(), 0, VK_WHOLE_SIZE };
            bufferInfos[2] = { batchBufC_[i]->handle(), 0, VK_WHOLE_SIZE };
            std::array<VkWriteDescriptorSet, 3> writes{};
            for (uint32_t b = 0; b < 3; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = batchSets_[i];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &bufferInfos[b];
            }
            vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        void* mapped = batchBufA_[i]->map(); std::memcpy(mapped, job.A, sizeA); batchBufA_[i]->unmap();
        mapped = batchBufB_[i]->map(); std::memcpy(mapped, job.B, sizeB); batchBufB_[i]->unmap();
    }

    // Record ALL dispatches into ONE command buffer.
    vkResetCommandBuffer(commandBuffer_, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer_, &beginInfo);
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto& job = jobs[i];
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                                 0, 1, &batchSets_[i], 0, nullptr);
        uint32_t pushConsts[3] = { (uint32_t)job.M, (uint32_t)job.K, (uint32_t)job.N };
        vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConsts), pushConsts);
        uint32_t groupsX = ((uint32_t)job.N + 15) / 16;
        uint32_t groupsY = ((uint32_t)job.M + 15) / 16;
        vkCmdDispatch(commandBuffer_, groupsX, groupsY, 1);
    }

    vkEndCommandBuffer(commandBuffer_);

    // ONE submit, ONE wait for the entire batch.
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;
    vkQueueSubmit(ctx_.computeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx_.computeQueue());

    vkFreeCommandBuffers(ctx_.device(), commandPool_, 1, &cmd);
    vkFreeDescriptorSets(ctx_.device(), descriptorPool_, 1, &descriptorSet);

    mapped = bufC.map(); std::memcpy(C, mapped, sizeC); bufC.unmap();
}

} // namespace stakml::vulkan