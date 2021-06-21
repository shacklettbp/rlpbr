#ifndef RLPBR_CORE_DEVICE_H_INCLUDED
#define RLPBR_CORE_DEVICE_H_INCLUDED

// TextureConstants "enum"
SHADER_CONST int TextureConstantsBaseOffset          = 0;
SHADER_CONST int TextureConstantsMROffset            = 1;
SHADER_CONST int TextureConstantsSpecularOffset      = 2;
SHADER_CONST int TextureConstantsNormalOffset        = 3;
SHADER_CONST int TextureConstantsEmittanceOffset     = 4;
SHADER_CONST int TextureConstantsTransmissionOffset  = 5;
SHADER_CONST int TextureConstantsClearcoatOffset     = 6;
SHADER_CONST int TextureConstantsAnisoOffset         = 7;
SHADER_CONST int TextureConstantsTexturesPerMaterial = 8;

// MaterialFlags "enum"
SHADER_CONST uint32_t MaterialFlagsComplex                = 1 << 0;
SHADER_CONST uint32_t MaterialFlagsThinWalled             = 1 << 1;
SHADER_CONST uint32_t MaterialFlagsHasBaseTexture         = 1 << 2;
SHADER_CONST uint32_t MaterialFlagsHasMRTexture           = 1 << 3;
SHADER_CONST uint32_t MaterialFlagsHasSpecularTexture     = 1 << 4;
SHADER_CONST uint32_t MaterialFlagsHasNormalMap           = 1 << 5;
SHADER_CONST uint32_t MaterialFlagsHasEmittanceTexture    = 1 << 6;
SHADER_CONST uint32_t MaterialFlagsHasTransmissionTexture = 1 << 7;
SHADER_CONST uint32_t MaterialFlagsHasClearcoatTexture    = 1 << 8;
SHADER_CONST uint32_t MaterialFlagsHasAnisotropicTexture  = 1 << 9;

#endif
