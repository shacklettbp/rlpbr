if (DEFINED ENV{OPTIX_ROOT})
    set(OPTIX_ROOT "${ENV{OPTIX_ROOT}}")
endif()

if (NOT DEFINED OPTIX_ROOT)
    message(FATAL_ERROR "Need to specify path to optix")
endif()

add_library(rlpbr_optix SHARED
    render.hpp render.cpp
    scene.hpp scene.cpp
    utils.hpp utils.cpp
)

target_link_libraries(rlpbr_optix rlpbr_backend CUDA::nvrtc)

target_include_directories(rlpbr_optix SYSTEM
    PUBLIC
        "${OPTIX_ROOT}/include"
)

set(NVRTC_COMPILE_FLAGS -std=c++17 -default-device -rdc true -arch compute_86
    -use_fast_math -lineinfo)

get_target_property(RLPBR_INCLUDES rlpbr_optix INCLUDE_DIRECTORIES)
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

target_compile_definitions(rlpbr_optix PRIVATE
    "-DNVRTC_OPTIONS=${NVRTC_OPTIONS}")
target_compile_definitions(rlpbr_optix PRIVATE
    "-DOPTIX_SHADER=${CMAKE_CURRENT_SOURCE_DIR}/shader.cu")
