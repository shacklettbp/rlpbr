add_definitions(-DGLM_FORCE_DEPTH_ZERO_TO_ONE)
SET(MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_library(rlpbr SHARED
    ../include/rlpbr.hpp rlpbr.cpp 
    ../include/rlpbr/config.hpp ../include/rlpbr/utils.hpp
    ../include/rlpbr/environment.hpp ../include/rlpbr/backend.hpp
    scene.hpp scene.cpp
    shader.hpp utils.hpp
    optix_render.hpp optix_render.cpp
    optix_scene.hpp optix_scene.cpp
    optix_utils.hpp optix_utils.cpp
)

target_include_directories(rlpbr
    PUBLIC 
        ${MAIN_INCLUDE_DIR}
)

target_include_directories(rlpbr SYSTEM
    PRIVATE
        /home/bps/rendering/optix/cur/include
)

target_link_libraries(rlpbr CUDA::cudart CUDA::nvrtc Threads::Threads glm)

set(NVRTC_COMPILE_FLAGS -std=c++17 -default-device -rdc true -arch compute_86
    -use_fast_math -lineinfo)

get_target_property(RLPBR_INCLUDES rlpbr INCLUDE_DIRECTORIES)
set(NVRTC_OPTIONS)
foreach(f ${RLPBR_INCLUDES})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"-I${f}\", ")
endforeach()
foreach(f ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"-I${f}\", ")
endforeach()
foreach(f ${NVRTC_COMPILE_FLAGS})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"${f}\", ")
endforeach()
set(NVRTC_OPTIONS "${NVRTC_OPTIONS}")

target_compile_definitions(rlpbr PRIVATE "-DNVRTC_OPTIONS=${NVRTC_OPTIONS}")
target_compile_definitions(rlpbr PRIVATE
    "-DOPTIX_SHADER=${CMAKE_CURRENT_SOURCE_DIR}/optix_shader.cu")

add_library(rlpbr_preprocess SHARED
    import.hpp import.cpp
    gltf.hpp gltf.cpp
    habitat_json.hpp habitat_json.cpp
    ../include/rlpbr/preprocess.hpp preprocess.cpp
)

target_link_libraries(rlpbr_preprocess rlpbr meshoptimizer simdjson)

# Add warnings as public properties on all targets defined here
get_property(ADDED_TARGETS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY BUILDSYSTEM_TARGETS)
foreach (LOCAL_TARGET ${ADDED_TARGETS})
    get_property(TARGET_TYPE TARGET ${LOCAL_TARGET} PROPERTY TYPE)
    if (${TARGET_TYPE} STREQUAL "SHARED_LIBRARY")
        target_compile_options(${LOCAL_TARGET} PUBLIC -Wall -Wextra -Wshadow)
    endif()
endforeach()
