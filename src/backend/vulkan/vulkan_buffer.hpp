#pragma once
#include <vulkan/vulkan.h>
#include <cstddef>

namespace stakml::vulkan {

class VulkanBuffer {
public:
    VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkDeviceSize size, VkBufferUsageFlags usage);
    ~VulkanBuffer();

    VkBuffer handle() const { return buffer_; }
    void* map();
    void unmap();
    VkDeviceSize size() const { return size_; }

private:
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props);

    VkDevice device_;
    VkPhysicalDevice physicalDevice_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_;
};

} // namespace stakml::vulkan