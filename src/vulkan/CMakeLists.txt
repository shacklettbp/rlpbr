add_definitions(-DVK_NO_PROTOTYPES)

# Hack to prefer linking against vulkan sdk
IF (DEFINED ENV{VULKAN_SDK})
    IF (DEFINED CMAKE_PREFIX_PATH)
        set(ORIG_PREFIX_PATH CMAKE_PREFIX_PATH)
    ENDIF()
    set(CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
ENDIF()

find_package(Vulkan REQUIRED)

IF (DEFINED ENV{VULKAN_SDK})
    IF (DEFINED ORIG_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH ORIG_PREFIX_PATH)
    ELSE()
        unset(CMAKE_PREFIX_PATH)
    endif()
ENDIF()

# Build vulkan dispatch table
get_directory_property(DIR_DEFINES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS )
set(DEFINE_STR "")
foreach (f ${DIR_DEFINES})
    list(APPEND DEFINE_STR "-D${f}")
endforeach()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dispatch")

add_custom_command(
    OUTPUT dispatch/dispatch.template
    COMMAND ${CMAKE_CXX_COMPILER} -x c -E -P -nostdinc ${DEFINE_STR} "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template" -o dispatch/dispatch.template
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template
)

add_custom_command(
    OUTPUT dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
           dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_vk_dispatch.py
            ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template dispatch
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_vk_dispatch.py
)

add_custom_target( generate_vk_dispatch DEPENDS
    dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
    dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
)

add_library(rlpbr_vulkan SHARED
    render.hpp render.cpp
    config.hpp
    core.hpp core.cpp
    cuda_interop.hpp cuda_interop.cpp
    descriptors.hpp descriptors.cpp descriptors.inl
    dispatch.hpp dispatch.cpp
    memory.hpp memory.cpp
    render.hpp render.cpp
    scene.hpp scene.cpp
    shader.hpp shader.cpp
    utils.hpp utils.cpp utils.inl
    present.hpp present.cpp
)

add_library(rlpbr_vulkan_headless SHARED
    linux_hacks.cpp
)

target_compile_definitions(rlpbr_vulkan PRIVATE
    "-DSHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/shaders/")

target_include_directories(rlpbr_vulkan
    PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/dispatch")
target_link_libraries(rlpbr_vulkan
    rlpbr_core Vulkan::Vulkan CUDA::cudart
    glslang spirv_reflect SPIRV glslang-default-resource-limits
    stb glfw)

add_dependencies(rlpbr_vulkan generate_vk_dispatch)
