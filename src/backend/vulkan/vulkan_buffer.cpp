#include "vulkan_buffer.hpp"
#include <stdexcept>

namespace stakml::vulkan {

VulkanBuffer::VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                           VkDeviceSize size, VkBufferUsageFlags usage)
    : device_(device), physicalDevice_(physicalDevice), size_(size) {

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, buffer_, &memReq);

    // Host-visible + coherent: on your APU this is genuinely cheap since
    // CPU and GPU share the same physical memory — no separate VRAM to copy across.
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, props);

    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory_) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate buffer memory");

    vkBindBufferMemory(device_, buffer_, memory_, 0);
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
    if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
}

void* VulkanBuffer::map() {
    void* data;
    vkMapMemory(device_, memory_, 0, size_, 0, &data);
    return data;
}

void VulkanBuffer::unmap() {
    vkUnmapMemory(device_, memory_);
}

uint32_t VulkanBuffer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

} // namespace stakml::vulkan