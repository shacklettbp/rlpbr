cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
cmake_policy(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindPkgConfig)

set(NAME rlpbr)

IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
ENDIF()

project(${NAME} LANGUAGES C CXX CUDA)

string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

if (NOT USE_BUNDLED)
    find_package(glm 0.9.9.8 QUIET)
endif()

# FIXME
find_package(glfw3 3.3 QUIET)
find_package(ZLIB QUIET)
pkg_check_modules(gtk3 QUIET IMPORTED_TARGET gtk+-3.0)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND gtk3_FOUND)
    set(SYS_GUI_FOUND ON)
else()
    set(SYS_GUI_FOUND OFF)
endif()

if (glfw3_FOUND AND ZLIB_FOUND AND SYS_GUI_FOUND AND NOT DISABLE_EDITOR)
    set(ENABLE_EDITOR ON)
else()
    message(STATUS "Not building editor")
    set(ENABLE_EDITOR OFF)
endif()

find_package(OpenImageIO QUIET)

add_subdirectory(external)

add_compile_definitions("RLPBR_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")
add_compile_options(-Wall -Wextra -Wshadow)

add_subdirectory(src)
add_subdirectory(bin)
