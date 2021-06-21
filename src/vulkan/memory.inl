namespace RLpbr {
namespace vk {

VkFormat MemoryAllocator::getTextureFormat(TextureFormat fmt)
{
    return texture_formats_[static_cast<uint32_t>(fmt)];
}

}
}
