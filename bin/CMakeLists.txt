add_executable(preprocess
    preprocess.cpp
)
target_link_libraries(preprocess rlpbr_preprocess)

add_executable(singlebench
    singlebench.cpp
)
target_link_libraries(singlebench rlpbr)

add_executable(save_frame
    save_frame.cpp
)
target_link_libraries(save_frame rlpbr)
