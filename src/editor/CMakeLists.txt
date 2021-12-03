include(FindPkgConfig)

find_package(ZLIB REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(editor_sys STATIC
    file_select.hpp file_select.cpp
)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    pkg_check_modules(gtk3 REQUIRED IMPORTED_TARGET gtk+-3.0)

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

add_executable(editor
    editor.hpp editor.cpp
    renderer.hpp renderer.cpp
    navmesh.hpp navmesh.cpp
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
    glm
    glfw
    RecastNavigation::Recast
    RecastNavigation::Detour
    meshoptimizer
    imgui_impl
    ZLIB::ZLIB
    simdjson
)
