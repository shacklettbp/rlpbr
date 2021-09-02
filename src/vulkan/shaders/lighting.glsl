#ifndef RLPBR_VK_LIGHTING_GLSL_INCLUDED
#define RLPBR_VK_LIGHTING_GLSL_INCLUDED

#include "math.glsl"
#include "inputs.glsl"
#include "utils.glsl"
#include "sampler.glsl"

// LightType "enum"
const uint32_t LightTypePoint = 0;
const uint32_t LightTypePortal = 1;
const uint32_t LightTypeEnvironment = 2;

struct PointLight {
    vec3 rgb;
    vec3 position;
};

struct PortalLight {
    vec3 corners[4];
};

struct LightSample {
    vec3 toLight;
    vec3 weight; // Irradiance / PDF
};

struct LightInfo {
    LightSample lightSample;
    vec3 shadowRayOrigin;
    float shadowRayLength;
};

PointLight unpackPointLight(vec4 data0, vec4 data1)
{
    return PointLight(
        vec3(data0.y, data0.z, data0.w),
        vec3(data1.x, data1.y, data1.z));
}

PortalLight unpackPortalLight(vec4 data1, vec4 data2, vec4 data3)
{
    PortalLight light = {{
        vec3(data1.x, data1.y, data1.z),
        vec3(data1.w, data2.x, data2.y),
        vec3(data2.z, data2.w, data3.x),
        vec3(data3.y, data3.z, data3.w),
    }};

    return light;
}

uint32_t unpackLight(in Environment env,
                     in uint32_t light_idx,
                     out PointLight point_light,
                     out PortalLight portal_light)
{
    PackedLight packed =
        lights[nonuniformEXT(env.baseLightOffset + light_idx)];

    vec4 data0 = packed.data[0];
    uint32_t light_type = floatBitsToUint(data0.x);

    vec4 data1 = packed.data[1];
    vec4 data2 = packed.data[2];
    vec4 data3 = packed.data[3];

    if (light_type == LightTypePoint) {
        point_light = unpackPointLight(data0, data1);
    } else if (light_type == LightTypePortal) {
        portal_light = unpackPortalLight(data1, data2, data3);
    }

    return light_type;
}

vec3 evalEnvMap(uint32_t map_idx, vec3 dir)
{
    vec2 uv = dirToLatLong(dir);

    vec3 v =
        textureLod(sampler2D(textures[map_idx], repeatSampler), uv, 0.0).xyz;

    return v;
}

LightSample sampleEnvMap(uint32_t map_idx,
    vec2 uv, float inv_selection_pdf)
{
    vec3 dir = octSphereMap(uv);

    vec3 irradiance = evalEnvMap(map_idx, dir);

    const float inv_pdf = 4.f * M_PI;

    LightSample light_sample;
    light_sample.toLight = dir;
    light_sample.weight = irradiance * inv_pdf * inv_selection_pdf;

    return light_sample;
}

vec3 getPortalLightPoint(PortalLight light, vec2 uv)
{
    vec3 upper = mix(light.corners[0], light.corners[3], uv.x);
    vec3 lower = mix(light.corners[1], light.corners[2], uv.x);

    return mix(lower, upper, uv.y);
}

LightSample samplePortal(
    PortalLight light, uint32_t map_idx,
    vec3 to_light, float inv_selection_pdf)
{
    float len2 = dot(to_light, to_light);
    vec3 dir = to_light * inversesqrt(len2);

    vec3 irradiance = evalEnvMap(map_idx, dir);

    // FIXME: redo this code to be faster
    vec3 side_a = light.corners[3] - light.corners[0];
    vec3 side_b = light.corners[1] - light.corners[0];

    const float area = length(side_a) * length(side_b);
    vec3 portal_normal = normalize(cross(side_a, side_b));

    float cos_theta = abs(dot(portal_normal, dir));

    float inv_pdf = (len2 < NEAR_ZERO) ? 0 : 
        (cos_theta * area / len2);

    LightSample light_sample;
    light_sample.toLight = dir;
    light_sample.weight = irradiance * inv_pdf * inv_selection_pdf;

    return light_sample;
}

void samplePointLight(
    PointLight light, vec3 origin, float inv_selection_pdf,
    out LightSample light_sample, out float dist_to_light2)
{
    vec3 to_light = light.position - origin;

    dist_to_light2 = dot(to_light, to_light);

    vec3 irradiance = light.rgb / dist_to_light2;

    light_sample.toLight = normalize(to_light);
    light_sample.weight = irradiance * inv_selection_pdf;
}

// Output functions

LightInfo sampleLights(inout Sampler rng, in Environment env, 
    in vec3 origin, in vec3 base_normal)
{
    //uint32_t total_lights = env.numLights + 1;
    uint32_t total_lights = env.numLights;// + 1;

    uint32_t light_idx = uint32_t(samplerGet1D(rng) * total_lights);

    vec2 light_sample_uv = samplerGet2D(rng);

    float inv_selection_pdf = float(total_lights);

    PointLight point_light = { vec3(0), vec3(0) };
    PortalLight portal_light = {{ vec3(0), vec3(0), vec3(0), vec3(0) }};
    uint32_t light_type = 0;

    if (light_idx < env.numLights) {
        light_type = unpackLight(env, light_idx, point_light, portal_light);
    } else {
        light_type = LightTypeEnvironment;
    }

    vec3 light_position = vec3(0); 
    vec3 dir_check = vec3(0);
    LightSample light_sample = LightSample(vec3(0), vec3(0));

    if (light_type == LightTypePoint) {
        light_position = point_light.position;
        dir_check = light_position - origin;
    } else if (light_type == LightTypePortal) {
        light_position = getPortalLightPoint(portal_light, light_sample_uv);
        dir_check = light_position - origin;
    } else {
        light_sample = sampleEnvMap(env.baseTextureOffset, light_sample_uv,
                                    inv_selection_pdf);
        dir_check = light_sample.toLight;
    }

    vec3 shadow_offset_normal =
        dot(dir_check, base_normal) > 0 ? base_normal : -base_normal;

    vec3 shadow_origin =
        offsetRayOrigin(origin, shadow_offset_normal);

    float shadow_len = 0;
    if (light_type == LightTypePoint) {
        float shadow_len2;
        samplePointLight(point_light, shadow_origin, inv_selection_pdf,
                         light_sample, shadow_len2);

        shadow_len = sqrt(shadow_len2);
    } else if (light_type == LightTypePortal) {
        vec3 to_light = light_position - shadow_origin;
        light_sample = samplePortal(portal_light, env.baseTextureOffset,
                                    to_light, inv_selection_pdf);

        shadow_len = LARGE_DISTANCE;
    } else {
        shadow_len = LARGE_DISTANCE;
    }

    LightInfo info = {
        light_sample,
        shadow_origin,
        shadow_len,
    };

    return info;
}


#endif
