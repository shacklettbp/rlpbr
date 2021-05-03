if (DEFINED ENV{OPTIX_ROOT})
    set(OPTIX_ROOT $ENV{OPTIX_ROOT})
endif()

if (NOT DEFINED OPTIX_ROOT)
    message(FATAL_ERROR "Need to specify path to optix")
endif()

add_library(rlpbr_optix SHARED
    render.hpp render.cpp
    scene.hpp scene.cpp
    utils.hpp utils.cpp cuda_utils.hpp
    physics.hpp physics.cpp #collisions.cu
    texture.hpp texture.cpp texture.inl
)

target_link_libraries(rlpbr_optix 
    PRIVATE rlpbr_core CUDA::nvrtc stb)

target_include_directories(rlpbr_optix SYSTEM
    PUBLIC
        "${OPTIX_ROOT}/include"
)

# Disable for now while collisions.cu disabled
#target_compile_options(rlpbr_optix PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math -rdc=true -lineinfo>)
#
#set_property(TARGET rlpbr_optix PROPERTY CUDA_ARCHITECTURES 86)
#set_property(TARGET rlpbr_optix PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(NVRTC_COMPILE_FLAGS -std=c++17 -default-device -rdc true
    -use_fast_math -lineinfo)

get_target_property(RLPBR_OPTIX_INCLUDES rlpbr_optix INCLUDE_DIRECTORIES)
get_target_property(RLPBR_CORE_INCLUDES rlpbr_core INCLUDE_DIRECTORIES)
set(NVRTC_OPTIONS)
foreach(f ${RLPBR_OPTIX_INCLUDES})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"-I${f}\", ")
endforeach()
foreach(f ${RLPBR_CORE_INCLUDES})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"-I${f}\", ")
endforeach()
foreach(f ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"-I${f}\", ")
endforeach()
foreach(f ${NVRTC_COMPILE_FLAGS})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"${f}\", ")
endforeach()
set(NVRTC_OPTIONS "${NVRTC_OPTIONS}")

target_compile_definitions(rlpbr_optix PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:NVRTC_OPTIONS=${NVRTC_OPTIONS}>)
target_compile_definitions(rlpbr_optix PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:OPTIX_SHADER=${CMAKE_CURRENT_SOURCE_DIR}/shader.cu>)
