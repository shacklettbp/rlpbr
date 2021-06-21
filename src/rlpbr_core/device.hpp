#pragma once

namespace RLpbr {

namespace Shader {

#define SHADER_CONST static constexpr
#include "device.h"
#undef SHADER_CONST

}

struct TextureConstants {
    static constexpr int baseOffset = Shader::TextureConstantsBaseOffset;
    static constexpr int mrOffset = Shader::TextureConstantsMROffset;
    static constexpr int specularOffset = Shader::TextureConstantsSpecularOffset;
    static constexpr int normalOffset = Shader::TextureConstantsNormalOffset;
    static constexpr int emittanceOffset = Shader::TextureConstantsEmittanceOffset;
    static constexpr int transmissionOffset = Shader::TextureConstantsTransmissionOffset;
    static constexpr int clearcoatOffset = Shader::TextureConstantsClearcoatOffset;
    static constexpr int anisoOffset = Shader::TextureConstantsAnisoOffset;
    static constexpr int numTexturesPerMaterial =
        Shader::TextureConstantsTexturesPerMaterial;
};

enum class MaterialFlags : uint32_t {
    Complex = Shader::MaterialFlagsComplex,
    ThinWalled = Shader::MaterialFlagsThinWalled,
    HasBaseTexture = Shader::MaterialFlagsHasBaseTexture,
    HasMRTexture = Shader::MaterialFlagsHasMRTexture,
    HasSpecularTexture = Shader::MaterialFlagsHasSpecularTexture,
    HasNormalMap = Shader::MaterialFlagsHasNormalMap,
    HasEmittanceTexture = Shader::MaterialFlagsHasEmittanceTexture,
    HasTransmissionTexture = Shader::MaterialFlagsHasTransmissionTexture,
    HasClearcoatTexture = Shader::MaterialFlagsHasClearcoatTexture,
    HasAnisotropicTexture = Shader::MaterialFlagsHasAnisotropicTexture,
};

}
