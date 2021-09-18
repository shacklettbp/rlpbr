include(FetchContent)

FetchContent_Declare(
  simdjson
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/simdjson
)

FetchContent_MakeAvailable(simdjson)

set(MESHOPT_BUILD_SHARED_LIBS ON)
add_subdirectory(meshoptimizer EXCLUDE_FROM_ALL)

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE stb)

add_library(libigl INTERFACE)
target_include_directories(libigl SYSTEM INTERFACE
    libigl/include
    eigen
)

add_library(mikktspace STATIC
    MikkTSpace/mikktspace.c 
    MikkTSpace/mikktspace.h)
target_include_directories(mikktspace SYSTEM INTERFACE
    MikkTSpace)

IF (NOT glm_FOUND OR USE_BUNDLED)
    message(STATUS "Using bundled GLM")
    add_library(glm INTERFACE)
    target_include_directories(glm SYSTEM
        INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")
ENDIF()

add_library(spirv_reflect SHARED
    SPIRV-Reflect/spirv_reflect.h SPIRV-Reflect/spirv_reflect.c)
target_include_directories(spirv_reflect
    INTERFACE SPIRV-Reflect)

add_subdirectory(glslang EXCLUDE_FROM_ALL)
target_include_directories(glslang SYSTEM
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glslang/glslang/Public>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glslang/StandAlone>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glslang/SPIRV>
)

set(OpenGL_GL_PREFERENCE "GLVND" CACHE INTERNAL "")

set(BUILD_SHARED_LIBS ON)

add_library(imgui INTERFACE)
target_include_directories(imgui INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
)
