#ifndef RLPBR_VK_SHADERS_EXPOSURE_COMMON_GLSL_INCLUDED
#define RLPBR_VK_SHADERS_EXPOSURE_COMMON_GLSL_INCLUDED

float avgWorkgroupIlluminance(float illuminance, bool oob)
{
    if (gl_SubgroupID == 0 && subgroupElect()) {
        workgroupCount = 0;
    }

    barrier();

    float denom = oob ? 0.f : 1.f;
    illuminance = oob ? 0.f : illuminance;

    illuminance = subgroupAdd(illuminance);
    denom = subgroupAdd(denom);

    if (subgroupElect()) {
        float subgroup_avg = 0.f;
        if (denom > 0.f) {
            atomicAdd(workgroupCount, 1);
            subgroup_avg = illuminance / denom;
        }

        workgroupScratch[gl_SubgroupID] = subgroup_avg;
    }

    barrier();

    if (gl_SubgroupID == 0) {
        float tmp = gl_SubgroupInvocationID < NUM_SUBGROUPS ?
            workgroupScratch[gl_SubgroupInvocationID] : 0;

        float full_avg = subgroupAdd(tmp) / float(workgroupCount);
        return full_avg;
    } else {
        return 0.f;
    }
}

void setExposureIlluminance(u32vec3 idx, float avg_illuminance)
{
    uint32_t subres_x = idx.x / NUM_WORKGROUPS_X;
    uint32_t subres_y = idx.y / NUM_WORKGROUPS_Y;
    uint32_t linear_idx = idx.z * NUM_WORKGROUPS_Y * NUM_WORKGROUPS_X +
        subres_y * NUM_WORKGROUPS_X + subres_x;

    if (gl_SubgroupID == 0 && subgroupElect()) {
        tonemapIlluminanceBuffer[linear_idx] += avg_illuminance;
    }
}

#endif
