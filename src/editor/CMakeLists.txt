set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(editor_sys STATIC
    file_select.hpp file_select.cpp
)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_compile_definitions(editor_sys PRIVATE
        USE_GTK
    )

    target_link_libraries(editor_sys PRIVATE
        PkgConfig::gtk3)
endif()

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/imgui")

add_library(imgui_impl STATIC
    imgui_extensions.hpp imgui_extensions.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.h
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.h
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

target_compile_definitions(imgui_impl
    PRIVATE
        "-DVK_NO_PROTOTYPES"
)

target_link_libraries(imgui_impl
    PUBLIC
        imgui
        glfw
)

add_library(navmesh_utils STATIC
    navmesh.hpp navmesh.cpp
)

target_link_libraries(navmesh_utils PUBLIC
    RecastNavigation::Recast
    RecastNavigation::Detour
    rlpbr_core
    rlpbr_vulkan
    meshoptimizer
)

add_executable(editor
    editor.hpp editor.cpp
    renderer.hpp renderer.cpp
    utils.hpp
    json.hpp json.cpp
)

target_compile_definitions(editor PRIVATE
    "EDITOR_SHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_link_libraries(editor PRIVATE
    editor_sys
    rlpbr_core
    rlpbr_vulkan
    navmesh_utils
    glm
    glfw
    imgui_impl
    ZLIB::ZLIB
    simdjson
)

string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda")
add_executable(stats
    stats.cpp stats.hpp stats.cu
)

target_link_libraries(stats PRIVATE
    rlpbr
    navmesh_utils
)

add_executable(datagen
    datagen.cpp ../../bin/oiio_bridge.cpp
)

target_link_libraries(datagen PRIVATE
    rlpbr
    navmesh_utils
    OpenImageIO
)
