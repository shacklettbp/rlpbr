add_library(rlpbr_preprocess SHARED
    gltf.hpp gltf.inl
    habitat_json.hpp habitat_json.inl
    import.hpp import.cpp
    ../../include/rlpbr/preprocess.hpp preprocess.hpp preprocess.cpp
    physics.hpp physics.inl
)

target_link_libraries(rlpbr_preprocess
    PRIVATE
        rlpbr_core
        meshoptimizer
        simdjson
        libigl
        mikktspace
)

target_include_directories(rlpbr_preprocess
    INTERFACE
        ${MAIN_INCLUDE_DIR}
)
