#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

#ifdef VALIDATE
#extension GL_EXT_debug_printf : enable
#endif

#define SHADER_CONST const
#include "rlpbr_core/device.h"
#undef SHADER_CONST

#include "shader_common.h"

#define NUM_BINS (130)

layout (set = 0, binding = 0, scalar) buffer TonemapIlluminance {
    float illuminanceBuffer[];
};

shared uint32_t histogramBins[NUM_BINS];

uint32_t divideRoundUp(uint32_t a, uint32_t b)
{
    return (a + (b - 1)) / b;
}

int computeBin(float luminance)
{
    // BIN_ZERO_THRESHOLD is chosen so 
    // log(BIN_ZERO_THRESHOLD - epsilon) == MIN_LOG_LUMINANCE
    // This stops bin 0 from being slightly overweighted
    float log_luminance = luminance < BIN_ZERO_THRESHOLD ?
        MIN_LOG_LUMINANCE : log2(luminance);

    float remapped = (log_luminance - MIN_LOG_LUMINANCE) * INV_LOG_LUMINANCE_RANGE;

    return min(int(remapped * NUM_BINS), NUM_BINS - 1);
}

layout (local_size_x = LOCAL_WORKGROUP_X,
        local_size_y = LOCAL_WORKGROUP_Y,
        local_size_z = LOCAL_WORKGROUP_Z) in;
void main()
{
    uint32_t batch_idx = gl_GlobalInvocationID.z;
    u32vec2 xy_idx = gl_LocalInvocationID.xy;

    if (gl_LocalInvocationIndex < NUM_BINS) {
        histogramBins[gl_LocalInvocationIndex] = 0.f;
    }

    u32vec2 illuminance_res = u32vec2(
        RES_X / LOCAL_WORKGROUP_X,
        RES_Y / LOCAL_WORKGROUP_Y);

    u32vec2 elems_per_thread = u32vec2(
            divideRoundUp(illuminance_res.x, LOCAL_WORKGROUP_X),
            divideRoundUp(illuminance_res.y, LOCAL_WORKGROUP_Y));

    if (xy_idx.x * elems_per_thread >= illuminance_res.x ||
        xy_idx.y * elems_per_thread >= illuminance_res.y) {
        return;
    }

    for (int y = 0; y < elems_per_thread.y; y++) {
        for (int x = 0; x < elems_per_thread.x; x++) {
            uint32_t y_idx = xy_idx.y * elems_per_thread.y + y;
            uint32_t x_idx = xy_idx.x * elems_per_thread.x + x;

            uint32_t idx = batch_idx * illuminance_res.x * illuminance_res.y +
                y_idx * illuminance_res.x + x_idx;

            float illuminance = illuminanceBuffer[idx];

            int bin_idx = computeBin(illuminance);

            atomicAdd(histogramBins[bin_idx], 1);
        }
    }

    barrier();

    if (gl_SubgroupID != 0) {
        return;
    }

    const int bins_per_thread = (NUM_BINS - 2) / SUBGROUP_SIZE;
    for (int i = 0; i < bins_per_thread; i++) {
        int cur_idx = gl_SubgroupInvocationID + SUBGROUP_SIZE * i + 1;
        int cur = histogramBins[cur_idx];

        int partial_sum = subgroupInclusiveAdd(cur);
        int prev_sum = i > 0 ? histogramBins[SUBGROUP_SIZE * i - 1] : 0;

        histogramBins[cur_idx] = partial_sum + prev_sum;
        subgroupBarrier();
    }

    int total = histogramBins[NUM_BINS - 2];

    int bin_idx = gl_LocalInvocationIndex;
    float histogram_val = histogramBins[bin_idx];

    atomicAdd(histogramBins[0], histogram_val);

}
