#ifndef RLPBR_VK_SHADERS_EXPOSURE_COMMON_GLSL_INCLUDED
#define RLPBR_VK_SHADERS_EXPOSURE_COMMON_GLSL_INCLUDED

shared float illuminanceScratch[NUM_SUBGROUPS];
shared uint32_t illuminanceCount;
void setExposureIlluminance(float illuminance, bool oob)
{
    if (gl_SubgroupID == 0 && subgroupElect()) {
        illuminanceCount = 0;
    }

    barrier();

    float denom = oob ? 0.f : 1.f;
    illuminance = oob ? 0.f : illuminance;

    illuminance = subgroupAdd(illuminance);
    denom = subgroupAdd(denom);

    if (subgroupElect()) {
        float subgroup_avg = 0.f;
        if (denom > 0.f) {
            atomicAdd(illuminanceCount, 1);
            subgroup_avg = illuminance / denom;
        }

        illuminanceScratch[gl_SubgroupID] = subgroup_avg;
    }

    barrier();

    if (gl_SubgroupID != 0) return;

    float tmp = gl_SubgroupInvocationID < NUM_SUBGROUPS ?
        illuminanceScratch[gl_SubgroupInvocationID] : 0;

    float full_avg = subgroupAdd(tmp) / float(illuminanceCount);

    if (subgroupElect()) {
        uint32_t linear_idx = gl_GlobalInvocationID.z *
            EXPOSURE_RES_X * EXPOSURE_RES_Y +
                gl_WorkGroupID.y * EXPOSURE_RES_X +
                gl_WorkGroupID.x;

        tonemapIlluminanceBuffer[linear_idx] = full_avg;
    }
    
}

#endif
