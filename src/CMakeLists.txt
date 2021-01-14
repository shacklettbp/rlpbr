add_definitions(-DGLM_FORCE_DEPTH_ZERO_TO_ONE)
SET(MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_library(rlpbr SHARED
    ../include/rlpbr.hpp rlpbr.cpp 
    ../include/rlpbr/config.hpp ../include/rlpbr/utils.hpp
    ../include/rlpbr/environment.hpp
    scene.hpp scene.cpp
    shader.hpp utils.hpp
)

target_include_directories(rlpbr
    PUBLIC 
        ${MAIN_INCLUDE_DIR}
)

target_link_libraries(rlpbr CUDA::cudart Threads::Threads glm)

add_library(rlpbr_preprocess SHARED
    import.hpp import.cpp
    gltf.hpp gltf.cpp
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
