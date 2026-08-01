#pragma once
#include "vulkan_context.hpp"
#include "vulkan_buffer.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace stakml::vulkan {

// One independent matmul job: C = A @ B, all row-major float32.
struct MatmulJob {
    const float* A;
    const float* B;
    float* C;
    size_t M, K, N;
};

class VulkanMatmul {
public:
    explicit VulkanMatmul(VulkanContext& ctx);
    ~VulkanMatmul();

    // Single matmul - one submission, one wait. Same as before.
    void run(const float* A, const float* B, float* C,
             size_t M, size_t K, size_t N);

    // Batched: records ALL jobs into ONE command buffer, submits ONCE,
    // waits ONCE. Each job gets its own buffer trio (A/B/C) — distinct
    // buffers per job, not the shared bufA_/bufB_/bufC_ used by run() -
    // because dispatches within one command buffer without barriers
    // between them may execute concurrently on the GPU, so jobs sharing
    // a buffer would race. This is the experiment for whether batching
    // amortizes the fixed per-submission stall that dominates at small
    // matmul sizes.
    void runBatch(const std::vector<MatmulJob>& jobs);

private:
    void createDescriptorSetLayout();
    void createPipeline();

    VulkanContext& ctx_;

    VkShaderModule shaderModule_        = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout_     = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_    = VK_NULL_HANDLE;
    VkPipeline pipeline_                = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_    = VK_NULL_HANDLE;
    VkCommandPool commandPool_          = VK_NULL_HANDLE;
};

} // namespace stakml::vulkan