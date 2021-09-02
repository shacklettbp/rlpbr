#ifndef RLPBR_VK_SAMPLER_GLSL_INCLUDED
#define RLPBR_VK_SAMPLER_GLSL_INCLUDED

uint32_t samplerSeedHash(uint32_t seed)
{
    return seed;
}

#ifdef ZSOBOL_SAMPLING

uint32_t mix32(uint32_t v)
{
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;

    return v;
}

u32vec2 mix32x2(const uint32_t v)
{
    uint64_t vbig = v;
    vbig ^= (vbig >> 31);
    vbig *= 0x7fb5d329728ea185l;
    vbig ^= (vbig >> 27);
    vbig *= 0x81dadef4bc2dd44dl;
    vbig ^= (vbig >> 33);

    return u32vec2(uint32_t(vbig >> 32),
                  uint32_t(vbig));
}

// Space out bottom 16 bits with 0s
uint32_t mortonSpace16(uint32_t v) {
    v = (v ^ (v << 8)) & 0x00ff00ff;
    v = (v ^ (v << 4)) & 0x0f0f0f0f;
    v = (v ^ (v << 2)) & 0x33333333;
    v = (v ^ (v << 1)) & 0x55555555;

    return v;
};

uint32_t morton2D32(uint32_t x, uint32_t y) {
    return (mortonSpace16(y) << 1) | mortonSpace16(x);
}

uint32_t zSobolMortonIndex(uint32_t x, uint32_t y, uint32_t s)
{
#ifdef ZSOBOL_ODD_POWER
    // Add extra bit for odd powers of 2 to round sample index up
    s <<= 1;
#endif

    // Weird idea: what would happen if you expanded the morton code
    // to 3D and got low discrepancy across the batch?
    uint32_t morton_2d = morton2D32(x, y);

    return (morton_2d << ZSOBOL_INDEX_SHIFT) | s;
}

uint32_t zSobolPermutationDigit(uint32_t p, uint32_t d)
{
    uint32_t a = 0xe4b4d878;
    uint32_t b = 0x6c9ce1b1;
    uint32_t c = 0x934b4ec6;
    
    uint32_t mid = p & 0x6;
    
    uint32_t t = p < 8 ? a : p < 16 ? b : c;
    
    uint32_t perm_set = bool(p & 1) ? bitfieldReverse(t) : t;
    
    uint32_t perm = perm_set >> (mid * 4);
    
    return (perm >> (d * 2)) & 3;
}

uint32_t zSobolHashMortonPrefix(uint32_t dim, uint32_t idx)
{
    return mix32(idx ^ (0x55555555 * dim));
}

uint32_t zSobolHashPermute(uint32_t dim, uint32_t idx)
{
    uint32_t hash = zSobolHashMortonPrefix(dim, idx);
    return (hash >> 24) % 24;
}

uint32_t zSobolCurrentSampleIndex(uint32_t dim, uint32_t morton_idx)
{
    uint32_t sample_idx = 0;

#ifdef ZSOBOL_ODD_POWER
    const int last_digit = 1;
#else
    const int last_digit = 0;
#endif

    for (int i = int(ZSOBOL_NUM_BASE4) - 1; i >= last_digit; --i) {
        int digit_shift = 2 * i;
        uint32_t digit = (morton_idx >> digit_shift) & 3;
        uint32_t p =
            zSobolHashPermute(dim, morton_idx >> (digit_shift + 2));
        uint32_t permuted = zSobolPermutationDigit(p, digit);
        sample_idx |= permuted << digit_shift;
    }

#ifdef ZSOBOL_ODD_POWER
     uint32_t final_digit = morton_idx & 3;
     sample_idx |= final_digit;
     sample_idx >>= 1;
     sample_idx ^= zSobolHashMortonPrefix(dim, morton_idx >> 2) & 1;
#endif

    return sample_idx;
}

float zSobolFinalize(uint32_t v, uint32_t hash_seed)
{
    const float nearest_one = 0.99999994;
    const float expNeg32 = 2.3283064e-10;

    v ^= hash_seed;
    return min(v * expNeg32, nearest_one);
}

const int zSobolMatrixSize = 32;

float zSobolSequenceDimZero(uint32_t sobol_idx, uint32_t hash_seed)
{
    uint32_t v = 0;

    for (int mat_idx = 0; mat_idx < zSobolMatrixSize;
         sobol_idx >>= 1, mat_idx++) {

        uint32_t cur_vec = 1 << (zSobolMatrixSize - 1 - mat_idx);

        if (bool(sobol_idx & 1)) v ^= cur_vec;
    }

    return zSobolFinalize(v, hash_seed);
}

float zSobolSequenceDimOne(uint32_t sobol_idx, uint32_t hash_seed)
{
    uint32_t v = 0;

    uint32_t prev_vec = 0x80000000;

    for (int mat_idx = 0; mat_idx < zSobolMatrixSize;
         sobol_idx >>= 1, mat_idx++) {

        uint32_t cur_vec = prev_vec;
        prev_vec = prev_vec ^ (prev_vec >> 1);

        if (bool(sobol_idx & 1)) v ^= cur_vec;
    }

    return zSobolFinalize(v, hash_seed);
}

struct Sampler {
    uint32_t seed;
    uint32_t mortonIdx;
    uint32_t dim;
};

Sampler makeSampler(uint32_t pixel_x, uint32_t pixel_y,
                    uint32_t sample_idx, uint32_t base_frame_idx)
{
    Sampler rng;
    rng.seed = samplerSeedHash(base_frame_idx);
    rng.mortonIdx = zSobolMortonIndex(pixel_x, pixel_y, sample_idx),
    rng.dim = 0;

    return rng;
}

float samplerGet1D(inout Sampler rng)
{
    uint32_t idx = zSobolCurrentSampleIndex(rng.dim, rng.mortonIdx);
    uint32_t hash_seed = mix32(rng.dim ^ rng.seed);
    rng.dim++;

    return zSobolSequenceDimZero(idx, hash_seed);
}

vec2 samplerGet2D(inout Sampler rng)
{
    uint32_t idx = zSobolCurrentSampleIndex(rng.dim, rng.mortonIdx);
    u32vec2 hash_seed = mix32x2(rng.dim ^ rng.seed);
    rng.dim += 2;

    return vec2(zSobolSequenceDimZero(idx, hash_seed.x),
                zSobolSequenceDimOne(idx, hash_seed.y));
}

#endif

#ifdef UNIFORM_SAMPLING

struct Sampler {
    uint32_t v;
};

Sampler makeSampler(uint32_t pixel_x, uint32_t pixel_y,
                    uint32_t sample_idx, uint32_t base_frame_idx)
{
    Sampler rng;

    rng.v = (pixel_y * RES_X + pixel_x) * SPP + sample_idx;

    uint32_t v1 = samplerSeedHash(base_frame_idx);
    uint32_t s0 = 0;

    for (int n = 0; n < 4; n++) {
        s0 += 0x9e3779b9;
        rng.v += ((v1 << 4) + 0xa341316c) ^
            (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((rng.v << 4) + 0xad90777d) ^
            (rng.v + s0) ^ ((rng.v >> 5) + 0x7e95761e);
    }

    return rng;
}

float samplerGet1D(inout Sampler rng)
{
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    rng.v = (LCG_A * rng.v + LCG_C);
    uint32_t next = rng.v & 0x00FFFFFF;

    return float(next) / float(0x01000000);
}

vec2 samplerGet2D(inout Sampler rng)
{
    return vec2(samplerGet1D(rng), samplerGet1D(rng));
}

#endif

#endif
