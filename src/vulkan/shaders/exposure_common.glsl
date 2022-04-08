#ifndef RLPBR_VK_SHADERS_EXPOSURE_COMMON_GLSL_INCLUDED
#define RLPBR_VK_SHADERS_EXPOSURE_COMMON_GLSL_INCLUDED

float avgWorkgroupIlluminance(float illuminance, bool oob)
{
    if (gl_SubgroupID == 0 && subgroupElect()) {
        workgroupCount = 0;
    }

    barrier();
    memoryBarrierShared();

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
    memoryBarrierShared();

    if (gl_SubgroupID == 0) {
        float tmp = gl_SubgroupInvocationID < NUM_SUBGROUPS ?
            workgroupScratch[gl_SubgroupInvocationID] : 0;

        float full_avg = subgroupAdd(tmp) / float(workgroupCount);
        return full_avg;
    } else {
        return 0.0;
    }
}

uint32_t getExposureIlluminanceIdx(u32vec3 idx)
{
    uint32_t subres_x = idx.x / LOCAL_WORKGROUP_X;
    uint32_t subres_y = idx.y / LOCAL_WORKGROUP_Y;
    return idx.z * NUM_WORKGROUPS_Y * NUM_WORKGROUPS_X +
        subres_y * NUM_WORKGROUPS_X + subres_x;
}

#ifdef ADAPTIVE_SAMPLING
float getPrevExposureIlluminance(u32vec3 idx)
{
    return tonemapIlluminanceBuffer[getExposureIlluminanceIdx(idx)];
}
#endif

void setExposureIlluminance(u32vec3 idx, float avg_illuminance)
{
    if (gl_SubgroupID == 0 && subgroupElect()) {
        tonemapIlluminanceBuffer[getExposureIlluminanceIdx(idx)] =
            avg_illuminance;
    }
}

#endif
