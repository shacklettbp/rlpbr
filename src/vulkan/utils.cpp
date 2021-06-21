#include "utils.hpp"

#include <cstdlib>
#include <iostream>

using namespace std;

namespace RLpbr {
namespace vk {

int exportBinarySemaphore(const DeviceState &dev, VkSemaphore semaphore)
{
    VkSemaphoreGetFdInfoKHR fd_info;
    fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.pNext = nullptr;
    fd_info.semaphore = semaphore;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    REQ_VK(dev.dt.getSemaphoreFdKHR(dev.hdl, &fd_info, &fd));

    return fd;
}

void printVkError(VkResult res, const char *msg)
{
#define ERR_CASE(val) \
    case VK_##val:    \
        cerr << #val; \
        break

    cerr << msg << ": ";
    switch (res) {
        ERR_CASE(NOT_READY);
        ERR_CASE(TIMEOUT);
        ERR_CASE(EVENT_SET);
        ERR_CASE(EVENT_RESET);
        ERR_CASE(INCOMPLETE);
        ERR_CASE(ERROR_OUT_OF_HOST_MEMORY);
        ERR_CASE(ERROR_OUT_OF_DEVICE_MEMORY);
        ERR_CASE(ERROR_INITIALIZATION_FAILED);
        ERR_CASE(ERROR_DEVICE_LOST);
        ERR_CASE(ERROR_MEMORY_MAP_FAILED);
        ERR_CASE(ERROR_LAYER_NOT_PRESENT);
        ERR_CASE(ERROR_EXTENSION_NOT_PRESENT);
        ERR_CASE(ERROR_FEATURE_NOT_PRESENT);
        ERR_CASE(ERROR_INCOMPATIBLE_DRIVER);
        ERR_CASE(ERROR_TOO_MANY_OBJECTS);
        ERR_CASE(ERROR_FORMAT_NOT_SUPPORTED);
        ERR_CASE(ERROR_FRAGMENTED_POOL);
        ERR_CASE(ERROR_UNKNOWN);
        ERR_CASE(ERROR_OUT_OF_POOL_MEMORY);
        ERR_CASE(ERROR_INVALID_EXTERNAL_HANDLE);
        ERR_CASE(ERROR_FRAGMENTATION);
        ERR_CASE(ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
        ERR_CASE(ERROR_SURFACE_LOST_KHR);
        ERR_CASE(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        ERR_CASE(SUBOPTIMAL_KHR);
        ERR_CASE(ERROR_OUT_OF_DATE_KHR);
        ERR_CASE(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        ERR_CASE(ERROR_VALIDATION_FAILED_EXT);
        ERR_CASE(ERROR_INVALID_SHADER_NV);
        ERR_CASE(ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
        ERR_CASE(ERROR_NOT_PERMITTED_EXT);
        ERR_CASE(ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
        default:
            cerr << "New vulkan error";
            break;
    }
    cerr << endl;
#undef ERR_CASE
}

}
}
