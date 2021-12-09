add_definitions(-DGLM_FORCE_DEPTH_ZERO_TO_ONE)
SET(MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_subdirectory(rlpbr_core)

add_subdirectory(preprocess)

# Backends
add_subdirectory(optix)
add_subdirectory(vulkan)

add_library(rlpbr SHARED
    ../include/rlpbr.hpp rlpbr.cpp 
)

target_link_libraries(rlpbr
    PUBLIC 
        rlpbr_core
        rlpbr_vulkan
        rlpbr_vulkan_headless
    INTERFACE
        CUDA::cudart
        glm
)

if (OPTIX_ENABLED)
    target_link_libraries(rlpbr PUBLIC rlpbr_optix)
    target_compile_definitions(rlpbr PRIVATE OPTIX_ENABLED)
endif()

target_include_directories(rlpbr
    INTERFACE
        ${MAIN_INCLUDE_DIR}
)

add_subdirectory(editor)

