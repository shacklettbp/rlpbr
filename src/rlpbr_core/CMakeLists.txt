add_library(rlpbr_core SHARED
    ${MAIN_INCLUDE_DIR}/rlpbr/config.hpp
    ${MAIN_INCLUDE_DIR}/rlpbr/utils.hpp
    ${MAIN_INCLUDE_DIR}/rlpbr/environment.hpp
    ${MAIN_INCLUDE_DIR}/rlpbr/backend.hpp
    scene.hpp scene.cpp
    utils.hpp
    physics.hpp
    device.hpp device.h
    common.hpp common.cpp
)

target_include_directories(rlpbr_core
    PUBLIC 
        ${MAIN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(rlpbr_core
    PUBLIC
        CUDA::cudart
        Threads::Threads
        glm
)
