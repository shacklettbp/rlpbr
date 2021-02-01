add_library(rlpbr_backend SHARED
    ${MAIN_INCLUDE_DIR}/rlpbr/config.hpp
    ${MAIN_INCLUDE_DIR}/rlpbr/utils.hpp
    ${MAIN_INCLUDE_DIR}/rlpbr/environment.hpp
    ${MAIN_INCLUDE_DIR}/rlpbr/backend.hpp
    scene.hpp scene.cpp
    shader.hpp utils.hpp
)

target_include_directories(rlpbr_backend
    PUBLIC 
        ${MAIN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(rlpbr_backend
    PUBLIC
        CUDA::cudart
        Threads::Threads
        glm
)
