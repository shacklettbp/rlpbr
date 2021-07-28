#ifndef RLPBR_VK_SHADERS_PACKED_H_INCLUDED
#define RLPBR_VK_SHADERS_PACKED_H_INCLUDED

// Packed Structs
struct PackedVertex {
    vec4 data[2];
};

struct PackedMaterial {
    u32vec4 data[2];
};

struct PackedMeshInfo {
    u32vec4 data;
};


#endif
