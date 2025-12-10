#pragma once

#include <vulkan/vulkan_core.h>
#include <iostream>
namespace veekay::graphics {

struct Buffer {
	VkBuffer buffer;
	VkDeviceMemory memory;
	void* mapped_region;

	Buffer(size_t size, const void* data,
	       VkBufferUsageFlags usage);
	~Buffer();

	void copy_to(const void* data, size_t size, size_t offset = 0) {
		if (mapped_region == nullptr) {
			std::cerr << "Buffer is not mapped!\n";
			return;
		}

		if (data == nullptr) {
			std::cerr << "Source data is null!\n";
			return;
		}

		// Копируем данные в mapped region с учетом offset
		std::memcpy(static_cast<char*>(mapped_region) + offset, data, size);
	}
};

struct Texture {
	uint32_t width;
	uint32_t height;
	VkFormat format;

	VkImage image;
	VkImageView view;
	VkDeviceMemory memory;

	Buffer* staging;

	Texture(VkCommandBuffer cmd,
	        uint32_t width, uint32_t height,
	        VkFormat format,
	        const void* pixels);
	~Texture();
};

} // namespace veekay::graphics
