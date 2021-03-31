set(MESHOPT_BUILD_SHARED_LIBS ON)
add_subdirectory(meshoptimizer EXCLUDE_FROM_ALL)

set(SIMDJSON_JUST_LIBRARY ON CACHE INTERNAL "")
set(SIMDJSON_BUILD_STATIC ON CACHE INTERNAL "")
add_subdirectory(simdjson EXCLUDE_FROM_ALL)

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
