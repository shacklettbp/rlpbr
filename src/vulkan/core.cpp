#include "core.hpp"

#include <vulkan/vulkan_core.h>
#include <rlpbr_core/utils.hpp>

#include "config.hpp"
#include "utils.hpp"

#include <csignal>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

using namespace std;

extern "C" {
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *,
                                                const VkAllocationCallbacks *,
                                                VkInstance *);

VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceLayerProperties(uint32_t *, VkLayerProperties *);
}

namespace RLpbr {
namespace vk {

static bool haveValidationLayers()
{
    uint32_t num_layers;
    REQ_VK(vkEnumerateInstanceLayerProperties(&num_layers, nullptr));

    DynArray<VkLayerProperties> layers(num_layers);

    REQ_VK(vkEnumerateInstanceLayerProperties(&num_layers, layers.data()));

    for (uint32_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        const auto &layer_prop = layers[layer_idx];
        if (!strcmp("VK_LAYER_KHRONOS_validation", layer_prop.layerName)) {
            return true;
        }
    }

    // FIXME check for VK_EXT_debug_utils

    cerr << "Validation layers unavailable" << endl;
    return false;
}

static VkInstance createInstance(bool enable_validation,
                                 const vector<const char *> &extra_exts)
{
    VkApplicationInfo app_info {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "RLpbr";
    app_info.pEngineName = "RLpbr";
    app_info.apiVersion = VK_API_VERSION_1_2;

    vector<const char *> layers;
    vector<const char *> extensions(extra_exts);

    vector<VkValidationFeatureEnableEXT> val_enabled;
    VkValidationFeaturesEXT val_features {};
    val_features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;

    if (enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);

        val_enabled.push_back(
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);

        char *best_practices = getenv("VK_BEST_VALIDATE");
        if (best_practices && best_practices[0] == '1') {
            val_enabled.push_back(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
        }

        char *gpu_debug_test = getenv("VK_GPU_VALIDATE");
        if (gpu_debug_test && gpu_debug_test[0] == '1') {
            val_enabled.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
        } else {
            val_enabled.push_back(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
            setenv("DEBUG_PRINTF_TO_STDOUT", "1", 1);
        }

        val_features.enabledValidationFeatureCount = val_enabled.size();
        val_features.pEnabledValidationFeatures = val_enabled.data();
    }

    VkInstanceCreateInfo inst_info {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = &val_features;
    inst_info.pApplicationInfo = &app_info;

    if (layers.size() > 0) {
        inst_info.enabledLayerCount = layers.size();
        inst_info.ppEnabledLayerNames = layers.data();
    }

    if (extensions.size() > 0) {
        inst_info.enabledExtensionCount = extensions.size();
        inst_info.ppEnabledExtensionNames = extensions.data();
    }

    VkInstance inst;
    REQ_VK(vkCreateInstance(&inst_info, nullptr, &inst));

    return inst;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
validationDebug(VkDebugUtilsMessageSeverityFlagBitsEXT,
                VkDebugUtilsMessageTypeFlagsEXT,
                const VkDebugUtilsMessengerCallbackDataEXT *data,
                void *)
{
    cerr << data->pMessage << endl;

    signal(SIGTRAP, SIG_IGN);
    raise(SIGTRAP);
    signal(SIGTRAP, SIG_DFL);

    return VK_FALSE;
}

static VkDebugUtilsMessengerEXT makeDebugCallback(VkInstance hdl,
                                                  const InstanceDispatch &dt)
{
    auto makeMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        dt.getInstanceProcAddr(hdl, "vkCreateDebugUtilsMessengerEXT"));

    assert(makeMessenger != nullptr);

    VkDebugUtilsMessengerEXT messenger;

    VkDebugUtilsMessengerCreateInfoEXT create_info {};
    create_info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = validationDebug;

    REQ_VK(makeMessenger(hdl, &create_info, nullptr, &messenger));

    return messenger;
}

InstanceState::InstanceState(bool enable_validation,
                             bool need_present,
                             const vector<const char *> &extra_exts)
    : hdl(createInstance(enable_validation && haveValidationLayers(),
                         extra_exts)),
      dt(hdl, need_present),
      debug_(enable_validation ? makeDebugCallback(hdl, dt) : VK_NULL_HANDLE)
{}

static void fillQueueInfo(VkDeviceQueueCreateInfo &info,
                          uint32_t idx,
                          const vector<float> &priorities)
{
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    info.queueFamilyIndex = idx;
    info.queueCount = priorities.size();
    info.pQueuePriorities = priorities.data();
}

VkPhysicalDevice InstanceState::findPhysicalDevice(
    const DeviceUUID &uuid) const
{
    uint32_t num_gpus;
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, nullptr));

    DynArray<VkPhysicalDevice> phys(num_gpus);
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, phys.data()));

    for (uint32_t idx = 0; idx < phys.size(); idx++) {
        VkPhysicalDevice phy = phys[idx];
        VkPhysicalDeviceIDProperties dev_id {};
        dev_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

        VkPhysicalDeviceProperties2 props {};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &dev_id;
        dt.getPhysicalDeviceProperties2(phy, &props);

        if (!memcmp(uuid.data(), dev_id.deviceUUID,
                    sizeof(DeviceUUID::value_type) * uuid.size())) {
            return phy;
        }
    }

    cerr << "Cannot find matching vulkan UUID" << endl;
    fatalExit();
}

DeviceState InstanceState::makeDevice(
    const DeviceUUID &uuid,
    uint32_t desired_gfx_queues,
    uint32_t desired_compute_queues,
    uint32_t desired_transfer_queues,
    add_pointer_t<VkBool32(VkInstance, VkPhysicalDevice, uint32_t)>
        present_check) const
{
    vector<const char *> extensions {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
        VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME,
    };

    bool need_present = present_check != nullptr;

    if (need_present) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    if (debug_ != VK_NULL_HANDLE) {
        extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }

    VkPhysicalDevice phy = findPhysicalDevice(uuid);

    VkPhysicalDeviceFeatures2 feats;
    feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats.pNext = nullptr;
    dt.getPhysicalDeviceFeatures2(phy, &feats);

    uint32_t num_queue_families;
    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                               nullptr);

    if (num_queue_families == 0) {
        cerr << "GPU doesn't have any queue families" << endl;
        fatalExit();
    }

    DynArray<VkQueueFamilyProperties2> queue_family_props(num_queue_families);
    for (auto &qf : queue_family_props) {
        qf.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        qf.pNext = nullptr;
    }

    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                               queue_family_props.data());

    // Currently only finds dedicated transfer, compute, and gfx queues
    // FIXME implement more flexiblity in queue choices
    optional<uint32_t> compute_queue_family;
    optional<uint32_t> gfx_queue_family;
    optional<uint32_t> transfer_queue_family;
    for (uint32_t i = 0; i < num_queue_families; i++) {
        const auto &qf = queue_family_props[i];
        auto &qf_prop = qf.queueFamilyProperties;

        if (!transfer_queue_family &&
            (qf_prop.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            transfer_queue_family = i;
        } else if (!compute_queue_family &&
                   (qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                   !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            compute_queue_family = i;
            ;
        } else if (!gfx_queue_family &&
                   (qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                   (!need_present || present_check(hdl, phy, i))) {
            gfx_queue_family = i;
        }

        if (transfer_queue_family && compute_queue_family &&
            gfx_queue_family) {
            break;
        }
    }

    if (!compute_queue_family || !gfx_queue_family || !transfer_queue_family) {
        cerr << "GPU does not support required separate queues" << endl;
        fatalExit();
    }

    const uint32_t num_gfx_queues =
        min(desired_gfx_queues, queue_family_props[*gfx_queue_family]
                                    .queueFamilyProperties.queueCount);
    const uint32_t num_compute_queues =
        min(desired_compute_queues, queue_family_props[*compute_queue_family]
                                        .queueFamilyProperties.queueCount);
    const uint32_t num_transfer_queues =
        min(desired_transfer_queues, queue_family_props[*transfer_queue_family]
                                         .queueFamilyProperties.queueCount);

    array<VkDeviceQueueCreateInfo, 3> queue_infos {};
    vector<float> gfx_pris(num_gfx_queues, VulkanConfig::gfx_priority);
    vector<float> compute_pris(num_compute_queues,
                               VulkanConfig::compute_priority);
    vector<float> transfer_pris(num_transfer_queues,
                                VulkanConfig::transfer_priority);
    fillQueueInfo(queue_infos[0], *gfx_queue_family, gfx_pris);
    fillQueueInfo(queue_infos[1], *compute_queue_family, compute_pris);
    fillQueueInfo(queue_infos[2], *transfer_queue_family, transfer_pris);

    VkDeviceCreateInfo dev_create_info {};
    dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_create_info.queueCreateInfoCount = 3;
    dev_create_info.pQueueCreateInfos = queue_infos.data();
    dev_create_info.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());
    dev_create_info.ppEnabledExtensionNames = extensions.data();

    dev_create_info.pEnabledFeatures = nullptr;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features {};
    accel_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel_features.pNext = nullptr;
    accel_features.accelerationStructure = true;

    VkPhysicalDeviceRayQueryFeaturesKHR rq_features {};
    rq_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rq_features.pNext = &accel_features;
    rq_features.rayQuery = true;

    VkPhysicalDeviceRobustness2FeaturesEXT robustness_features {};
    robustness_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;
    robustness_features.pNext = &rq_features;
    robustness_features.nullDescriptor = true;

    VkPhysicalDeviceLineRasterizationFeaturesEXT line_features {};
    line_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT;
    line_features.pNext = &robustness_features;
    line_features.smoothLines = true;

    VkPhysicalDeviceVulkan12Features vk12_features {};
    vk12_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12_features.pNext = &line_features;
    vk12_features.bufferDeviceAddress = true;
    vk12_features.descriptorIndexing = true;
    vk12_features.descriptorBindingPartiallyBound = true;
    vk12_features.descriptorBindingUpdateUnusedWhilePending = true;
    vk12_features.runtimeDescriptorArray = true;
    vk12_features.shaderStorageBufferArrayNonUniformIndexing = true;
    vk12_features.shaderSampledImageArrayNonUniformIndexing = true;
    vk12_features.shaderFloat16 = true;
    vk12_features.shaderInt8 = true;
    vk12_features.storageBuffer8BitAccess = true;

    VkPhysicalDeviceVulkan11Features vk11_features {};
    vk11_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vk11_features.pNext = &vk12_features;
    vk11_features.storageBuffer16BitAccess = true;

    VkPhysicalDeviceFeatures2 requested_features {};
    requested_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    requested_features.pNext = &vk11_features;

    requested_features.features.samplerAnisotropy = false;
    requested_features.features.shaderInt16 = true;
    requested_features.features.shaderInt64 = true;
    requested_features.features.wideLines = true;
    requested_features.features.fillModeNonSolid = true;

    dev_create_info.pNext = &requested_features;

    VkDevice dev;
    REQ_VK(dt.createDevice(phy, &dev_create_info, nullptr, &dev));

    return DeviceState {*gfx_queue_family,
                        *compute_queue_family,
                        *transfer_queue_family,
                        num_gfx_queues,
                        num_compute_queues,
                        num_transfer_queues,
                        phy,
                        dev,
                        DeviceDispatch(dev, need_present, true)};
}

}
}
