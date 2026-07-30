#pragma once
#include "vulkan_context.hpp"
#include <cstddef>

namespace stakml::vulkan {

// Dispatches C = A @ B (row-major float32) via a compute shader.
// Owns the pipeline + descriptor set layout so repeated matmul() calls
// don't rebuild Vulkan objects every time — those are the expensive,
// reusable-across-calls pieces (unlike the buffers, which are per-call).
class VulkanMatmul {
public:
    explicit VulkanMatmul(VulkanContext& ctx);
    ~VulkanMatmul();

    // A: M x K, B: K x N, C: M x N — all row-major, C pre-allocated by caller.
    void run(const float* A, const float* B, float* C,
             size_t M, size_t K, size_t N);

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