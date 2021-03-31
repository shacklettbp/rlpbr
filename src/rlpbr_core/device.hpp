#pragma once

namespace RLpbr {

struct TextureConstants {
    static constexpr int baseOffset = 0;
    static constexpr int mrOffset = 1;
    static constexpr int specularOffset = 2;
    static constexpr int normalOffset = 3;
    static constexpr int emittanceOffset = 4;
    static constexpr int transmissionOffset = 5;
    static constexpr int clearcoatOffset = 6;
    static constexpr int anisoOffset = 7;
    static constexpr int numTexturesPerMaterial = 8;
};

enum class MaterialFlags : uint32_t {
    Complex = 1 << 0,
    ThinWalled = 1 << 1,
    HasBaseTexture = 1 << 2,
    HasMRTexture = 1 << 3,
    HasSpecularTexture = 1 << 4,
    HasNormalMap = 1 << 5,
    HasEmittanceTexture = 1 << 6,
    HasTransmissionTexture = 1 << 7,
    HasClearcoatTexture = 1 << 8,
    HasAnisotropicTexture = 1 << 9,
};

}
