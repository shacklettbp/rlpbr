
namespace RLpbr {
namespace vk {

uint32_t getWorkgroupSize(uint32_t num_items)
{
    return (num_items + VulkanConfig::compute_workgroup_size - 1) /
           VulkanConfig::compute_workgroup_size;
}

}
}
