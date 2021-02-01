add_definitions(-DGLM_FORCE_DEPTH_ZERO_TO_ONE)
SET(MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_subdirectory(rlpbr_backend)

add_subdirectory(preprocess)

# Backends
add_subdirectory(optix)

add_library(rlpbr SHARED
    ../include/rlpbr.hpp rlpbr.cpp 
)

target_link_libraries(rlpbr
    PRIVATE
        rlpbr_backend
        rlpbr_optix
    INTERFACE
        CUDA::cudart
        glm
)

target_include_directories(rlpbr
    INTERFACE
        ${MAIN_INCLUDE_DIR}
)
