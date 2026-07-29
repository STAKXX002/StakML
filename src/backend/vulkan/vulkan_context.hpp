#pragma once
#include <vulkan/vulkan.h>
#include <optional>

namespace stakml::vulkan {

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VkDevice device() const { return device_; }
    VkQueue computeQueue() const { return computeQueue_; }
    uint32_t computeQueueFamily() const { return computeQueueFamily_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }

private:
    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamily_ = 0;
};

} // namespace stakml::vulkan