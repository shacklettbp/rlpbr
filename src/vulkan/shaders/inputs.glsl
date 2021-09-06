#ifndef RLPBR_VK_INPUTS_GLSL_INCLUDED
#define RLPBR_VK_INPUTS_GLSL_INCLUDED

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

// Unpacked Structs
struct Camera {
    vec3 origin;
    vec3 view;
    vec3 up;
    vec3 right;
    float rightScale;
    float upScale;
};

struct Vertex {
    vec3 position;
    vec3 normal;
    vec4 tangentAndSign;
    vec2 uv;
};

struct Triangle {
    Vertex a;
    Vertex b;
    Vertex c;
};

struct TangentFrame {
    vec3 tangent;
    vec3 bitangent;
    vec3 normal;
};

struct Environment {
    uint32_t sceneID;
    uint32_t baseMaterialOffset;
    uint32_t baseLightOffset;
    uint32_t numLights;
    uint64_t tlasAddr;
    uint32_t baseTextureOffset;
};

struct MeshInfo {
    uint32_t indexOffset;
};

#endif
