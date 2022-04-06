add_executable(preprocess
    preprocess.cpp
)
target_link_libraries(preprocess rlpbr_preprocess)

add_executable(singlebench
    singlebench.cpp
)
target_link_libraries(singlebench rlpbr)

add_executable(load_scene
    load_scene.cpp)
target_link_libraries(load_scene rlpbr)

if (OpenImageIO_FOUND)
    add_executable(save_frame
        save_frame.cpp
        oiio_bridge.hpp oiio_bridge.cpp
    )
    target_link_libraries(save_frame rlpbr OpenImageIO OpenImageIO_Util)
endif()

add_executable(make_sequence
    make_sequence.cpp
)
target_link_libraries(make_sequence rlpbr stb)

#add_executable(load_test
#    load_test.cpp
#)
#
#target_link_libraries(load_test rlpbr Threads::Threads)

find_package(GLEW QUIET)

if (glfw3_FOUND AND GLEW_FOUND)
    add_executable(fly
        fly.cpp
    )

    target_link_libraries(fly rlpbr glfw GLEW GL)
endif()
