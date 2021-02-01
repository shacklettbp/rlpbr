add_library(rlpbr_preprocess SHARED
    import.hpp import.cpp
    gltf.hpp gltf.cpp
    habitat_json.hpp habitat_json.cpp
    ../../include/rlpbr/preprocess.hpp preprocess.cpp
)

target_link_libraries(rlpbr_preprocess
    PRIVATE
        rlpbr_backend
        meshoptimizer
        simdjson
)

target_include_directories(rlpbr_preprocess
    INTERFACE
        ${MAIN_INCLUDE_DIR}
)
