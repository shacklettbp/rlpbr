#pragma once

#include "math_utils.cuh"

namespace RLpbr {
namespace optix {

inline uint32_t seedHash(uint32_t seed)
{
    return seed;
}

}
}

#ifdef ZSOBOL_SAMPLING
#include "sobol.cuh"

namespace RLpbr {
namespace optix {

// http://abdallagafar.com/publications/zsampler/
// Code is heavily based on ZSobolSampler from pbrt-v4
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.

class Sampler {
public:
    inline Sampler(uint32_t pixel_x, uint32_t pixel_y,
                   uint32_t sample_idx, uint32_t base_frame_idx)
        : seed_(seedHash(base_frame_idx)),
          dim_(0),
          morton_idx_(
              computeMortonIdx(pixel_x, pixel_y, sample_idx))
    {}

    inline float get1D()
    {
        uint32_t idx = curSampleIndex();
        uint32_t hash_seed = mix32(dim_ ^ seed_);
        dim_++;

        return finalize(sobol::sample(idx, 0), hash_seed);
    }

    inline float2 get2D()
    {
        uint32_t idx = curSampleIndex();
        uint2 hash_seed = mix32x2(dim_ ^ seed_);
        dim_ += 2;

        return make_float2(finalize(sobol::sample(idx, 0), hash_seed.x),
                           finalize(sobol::sample(idx, 1), hash_seed.y));
    }

private:
    inline uint32_t curSampleIndex() const
    {
        auto hashPermute = [this](uint32_t idx) {
            // PBRT version
            return (mix32(idx ^ (0x55555555 * dim_)) >> 24) % 24;

            // ZSampler paper version
            //constexpr int BITS = 24;
            //constexpr uint32_t MASK = (1 << BITS) - 1;
            //constexpr uint32_t Z = 0x9e377A;

            //idx ^= dim_ * 0x555555;
            //uint32_t x = (idx * Z) & MASK;
            //return (x * 24) >> BITS;
        };

        static const uint8_t permutations[24][4] = {
            {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1},
            {0, 3, 2, 1}, {0, 3, 1, 2}, {1, 0, 2, 3}, {1, 0, 3, 2},
            {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 2, 0}, {1, 3, 0, 2},
            {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 0, 1, 3}, {2, 0, 3, 1},
            {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 1, 2, 0}, {3, 1, 0, 2},
            {3, 2, 1, 0}, {3, 2, 0, 1}, {3, 0, 2, 1}, {3, 0, 1, 2},
        };

        uint32_t sample_idx = 0;

        constexpr int last_digit = isOddPower2 ? 1 : 0;

        for (int i = numIndexDigitsBase4 - 1; i >= last_digit; --i) {
            int digit_shift = 2 * i;
            int digit = (morton_idx_ >> digit_shift) & 3;
            int p = hashPermute(morton_idx_ >> (digit_shift + 2));
            uint32_t permuted = permutations[p][digit];
            sample_idx |= permuted << digit_shift;
        }

        if constexpr (isOddPower2) {
            int final_digit = morton_idx_ & 3;
            int p = hashPermute(morton_idx_ >> 2);
            sample_idx |= p & 3;
            sample_idx >>= 1;
        }
        return sample_idx;
    }

    static __forceinline__ float finalize(uint32_t v, uint32_t hash_seed)
    {
        // Initial reverse not necessary because sobol numbers are flipped

        // https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
        v += v << 2;
        v ^= v * 0xfe9b5742;
        v += hash_seed;
        v *= hash_seed | 1;

        // Laine - Karras hash
        //v += hash_seed;
        //v ^= v * 0x6c50b47cu;
        //v ^= v * 0xb82f1e52u;
        //v ^= v * 0xc7afe638u;
        //v ^= v * 0x8d22f6e6u;

        v = __brev(v);

        return min(v * 0x1p-32f, Constants::nearestOne);
    }

    static inline uint32_t computeMortonIdx(uint32_t x, uint32_t y, uint32_t s)
    {
        // Add extra bit for odd powers of 2 to round sample index up
        constexpr uint32_t index_shift = isOddPower2 ? (log2SPP + 1) : log2SPP;
        if constexpr (isOddPower2) {
            s <<= 1;
        }

        // Space out bottom 16 bits with 0s
        auto space16 = [](uint32_t v) {
            v = (v ^ (v << 8)) & 0x00ff00ff;
            v = (v ^ (v << 4)) & 0x0f0f0f0f;
            v = (v ^ (v << 2)) & 0x33333333;
            v = (v ^ (v << 1)) & 0x55555555;
            return v;
        };

        // Weird idea: what would happen if you expanded the morton code
        // to 3D and got low discrepancy across the batch?
        uint32_t morton_2d = (space16(y) << 1) | space16(x);

        return (morton_2d << index_shift) | s;
    }

    static constexpr uint32_t log2SPP = log2DownConst(SPP);

    static constexpr bool isOddPower2 = log2SPP & 1;

    // Number of base 4 digits for x, y coords + samples
    // Equals # base 2 digits for max dimension * 2 [X + Y] / 2 [base 4]
    // Plus the rounded up log base 4 of the SPP
    static constexpr uint32_t numIndexDigitsBase4 = 
        log2DownConst(maxConst(RES_X, RES_Y) - 1) + 1 + (log2SPP + 1) / 2;

    static_assert(numIndexDigitsBase4 * 2 <= 32,
                  "Not enough bits for morton code");

    const uint32_t seed_;
    const uint32_t morton_idx_;
    uint32_t dim_;
};

}
}

#endif

#ifdef OWEN_IDX_SAMPLING

namespace RLpbr {
namespace optix {

class Sampler {
public:
    inline Sampler(const uint3 &pixel_coords, uint32_t sample_idx,
                   uint32_t base_frame_idx)
        : v_((pixel_coords.y * RES_X + pixel_coords.x) * SPP + sample_idx)
    {
        uint v1 = seedHash(base_frame_idx + pixel_coords.z);
        uint s0 = 0;

        for (int n = 0; n < 4; n++) {
            s0 += 0x9e3779b9;
            v_ += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
            v1 += ((v_<<4)+0xad90777d)^(v_+s0)^((v_>>5)+0x7e95761e);
        }
    }

    inline float get1D()
    {
        return (float)next() / (float)0x01000000;
    }

    inline float2 get2D()
    {
        return make_float2(get1D(), get1D());
    }

private:
    inline uint next()
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        v_ = (LCG_A * v_ + LCG_C);
        return v_ & 0x00FFFFFF;
    }

    uint v_;
};

}
}

#endif

#ifdef UNIFORM_SAMPLING

namespace RLpbr {
namespace optix {

class Sampler {
public:
    inline Sampler(uint32_t pixel_x, uint32_t pixel_y,
                   uint32_t sample_idx, uint32_t base_frame_idx)
        : v_((pixel_y * RES_X + pixel_x) * SPP + sample_idx)
    {
        uint v1 = seedHash(base_frame_idx);
        uint s0 = 0;

        for (int n = 0; n < 4; n++) {
            s0 += 0x9e3779b9;
            v_ += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
            v1 += ((v_<<4)+0xad90777d)^(v_+s0)^((v_>>5)+0x7e95761e);
        }
    }

    inline float get1D()
    {
        return (float)next() / (float)0x01000000;
    }

    inline float2 get2D()
    {
        return make_float2(get1D(), get1D());
    }

private:
    inline uint next()
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        v_ = (LCG_A * v_ + LCG_C);
        return v_ & 0x00FFFFFF;
    }

    uint v_;
};

}
}

#endif
