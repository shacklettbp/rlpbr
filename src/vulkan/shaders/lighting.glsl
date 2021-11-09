#ifndef RLPBR_VK_LIGHTING_GLSL_INCLUDED
#define RLPBR_VK_LIGHTING_GLSL_INCLUDED

#include "math.glsl"
#include "inputs.glsl"
#include "utils.glsl"
#include "sampler.glsl"

// LightType "enum"
const uint32_t LightTypeSphere = 0;
const uint32_t LightTypeTriangle = 1;
const uint32_t LightTypePortal = 2;
const uint32_t LightTypeEnvironment = 3;

struct SphereLight {
    vec3 position;
    float radius;
    uint32_t matIdx;
};

struct TriangleLight {
    vec3 verts[3];
    uint32_t matIdx;
};

struct PortalLight {
    vec3 corners[4];
};

struct LightSample {
    vec3 toLight;
    vec3 irradiance;
    float pdf;
};

struct LightInfo {
    LightSample lightSample;
    vec3 shadowRayOrigin;
    float shadowRayLength;
};

struct DeltaLightInfo {
    vec3 position;
    vec3 irradiance;
};

SphereLight unpackSphereLight(VertRef vert_addr, vec4 data)
{
    uint32_t vert_idx = floatBitsToUint(data.y);
    vec3 position = unpackVertexPosition(vert_addr, vert_idx);

    return SphereLight(
        position,
        data.w,
        floatBitsToUint(data.z));
}

TriangleLight unpackTriangleLight(VertRef vert_addr, IdxRef idx_addr, vec4 data)
{
    u32vec3 indices = fetchTriangleIndices(idx_addr,
                                           floatBitsToUint(data.y));

    vec3 a = unpackVertexPosition(vert_addr, indices.x);
    vec3 b = unpackVertexPosition(vert_addr, indices.y);
    vec3 c = unpackVertexPosition(vert_addr, indices.z);
    uint32_t mat_idx = floatBitsToUint(data.z);

    TriangleLight light = {
        { a, b, c },
        mat_idx,
    };

    return light;
}

PortalLight unpackPortalLight(VertRef vert_addr, IdxRef idx_addr, vec4 data)
{
    uint32_t idx_offset = floatBitsToUint(data.y);
    u32vec4 indices = u32vec4(
        idx_addr[nonuniformEXT(idx_offset)].idx,
        idx_addr[nonuniformEXT(idx_offset + 1)].idx,
        idx_addr[nonuniformEXT(idx_offset + 2)].idx,
        idx_addr[nonuniformEXT(idx_offset + 3)].idx);

    vec3 a = unpackVertexPosition(vert_addr, indices.x);
    vec3 b = unpackVertexPosition(vert_addr, indices.y);
    vec3 c = unpackVertexPosition(vert_addr, indices.z);
    vec3 d = unpackVertexPosition(vert_addr, indices.w);

    PortalLight light = {{
        a,
        b,
        c,
        d,
    }};

    return light;
}

uint32_t unpackLight(in Environment env,
                     in VertRef vert_addr,
                     in IdxRef idx_addr,
                     in uint32_t light_idx,
                     out SphereLight sphere_light,
                     out TriangleLight tri_light,
                     out PortalLight portal_light)
{
    PackedLight packed =
        lights[nonuniformEXT(env.baseLightOffset + light_idx)];

    vec4 data = packed.data;
    uint32_t light_type = floatBitsToUint(data.x);

    if (light_type == LightTypeSphere) {
        sphere_light = unpackSphereLight(vert_addr, data);
    } else if (light_type == LightTypeTriangle) {
        tri_light = unpackTriangleLight(vert_addr, idx_addr, data);
    } else if (light_type == LightTypePortal) {
        portal_light = unpackPortalLight(vert_addr, idx_addr, data);
    } 

    return light_type;
}

vec3 evalEnvMap(uint32_t map_idx, vec3 dir)
{
    return vec3(0.f);

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
    light_sample.irradiance = irradiance;
    light_sample.pdf = 1.f / (inv_pdf * inv_selection_pdf);

    return light_sample;
}

vec3 getTriangleLightPoint(TriangleLight light, vec2 uv)
{
    float su0 = sqrt(uv.x);
    vec2 b = vec2(1.f - su0, uv.y * su0);
    vec3 barys = vec3(1.f - b.x - b.y, b.x, b.y);

    vec3 position = barys.x * light.verts[0] +
        barys.y * light.verts[1] +
        barys.z * light.verts[2];

    return position;
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
    light_sample.irradiance = irradiance;
    light_sample.pdf = 1.f / (inv_pdf * inv_selection_pdf);

    return light_sample;
}

void sampleSphereLight(
    SphereLight light, MatRef mat_addr,
    vec3 origin, float inv_selection_pdf, vec2 uv,
    out LightSample light_sample, out float dist_to_light)
{
    // Code from pbrt-v3
    // Compute coordinate system for sphere sampling
    float dc = distance(origin, light.position);
    float inv_dc = 1.f / dc;
    vec3 wc = (light.position - origin) * inv_dc;

    vec3 wc_x, wc_y;
    if (abs(wc.x) > abs(wc.y)) {
        wc_x = vec3(-wc.z, 0.f, wc.x) * inversesqrt(wc.x * wc.x + wc.z * wc.z);
    } else {
        wc_x = vec3(0.f, wc.z, -wc.y) * inversesqrt(wc.y * wc.y + wc.z * wc.z);
    }
    wc_y = cross(wc, wc_x);

    // Compute $\theta$ and $\phi$ values for sample in cone
    float sin_theta_max = light.radius * inv_dc;
    float sin_theta_max2 = sin_theta_max * sin_theta_max;
    float inv_sin_theta_max = 1.f / sin_theta_max;
    float cos_theta_max = sqrt(max(0.f, 1.f - sin_theta_max2));

    float cos_theta  = (cos_theta_max - 1.f) * uv.x + 1.f;
    float sin_theta2 = 1.f - cos_theta * cos_theta;

    if (sin_theta_max2 < 0.00068523f /* sin^2(1.5 deg) */) {
        /* Fall back to a Taylor series expansion for small angles, where
           the standard approach suffers from severe cancellation errors */
        sin_theta2 = sin_theta_max2 * uv.x;
        cos_theta = sqrt(1.f - sin_theta2);
    }

    // Compute angle $\alpha$ from center of sphere to sampled point on surface
    float cos_alpha = sin_theta2 * inv_sin_theta_max +
        cos_theta * sqrt(max(0.f, 1.f - sin_theta2 * inv_sin_theta_max *
                             inv_sin_theta_max));

    float sin_alpha = sqrt(max(0.f, 1.f - cos_alpha * cos_alpha));
    float phi = uv.y * 2.f * M_PI;;

    // Compute surface normal and sampled point on sphere
    vec3 normal =
        sin_alpha * cos(phi) * -wc_x +
        sin_alpha * sin(phi) * -wc_y +
        cos_alpha * -wc;

    vec3 light_point = light.position + light.radius * normal;

    // Uniform cone PDF.
    float inv_pdf = 2.f * M_PI * (1.f - cos_theta_max);

    vec3 to_light = light_point - origin;

    dist_to_light = length(to_light);

    vec3 emittance = getMaterialEmittance(mat_addr, light.matIdx);

    if (dist_to_light > 0.f) {
        light_sample.toLight = to_light / dist_to_light;
        light_sample.irradiance = emittance;
        light_sample.pdf = 1.f / (inv_pdf * inv_selection_pdf);
    } else {
        light_sample.toLight = vec3(0.f);
        light_sample.irradiance = vec3(0.f);
        light_sample.pdf = 0.f;
    }
}

float pdfTriangleLight(float inv_selection_pdf, vec3 to_light,
                       float tri_area, vec3 geo_normal)
{
    float cos_theta = abs(dot(normalize(to_light), geo_normal));

    float dist_to_light2 = dot(to_light, to_light);

    if (dist_to_light2 > 0.f && cos_theta > 0.f) {
        return dist_to_light2 / (tri_area * cos_theta * inv_selection_pdf);
    } else {
        return 0.f;
    }
}

void sampleTriangleLight(in TriangleLight tri_light,
                         in MatRef mat_addr,
                         in vec3 origin,
                         in vec3 sampled_position,
                         in float inv_selection_pdf,
                         out LightSample light_sample,
                         out float dist_to_light)
{
    vec3 c = cross(tri_light.verts[1] - tri_light.verts[0],
                   tri_light.verts[2] - tri_light.verts[0]);

    vec3 n = normalize(c);
    float area = 0.5f * length(c);

    vec3 to_light = sampled_position - origin;
    
    float dist_to_light2 = dot(to_light, to_light);
    dist_to_light = sqrt(dist_to_light2);
    to_light /= dist_to_light;

    vec3 emittance = getMaterialEmittance(mat_addr, tri_light.matIdx);

    float cos_theta = abs(dot(to_light, n));

    float pdf = dist_to_light2 /
        (area * cos_theta * inv_selection_pdf);

    if (dist_to_light > 0.f && cos_theta > 0.f) {
        light_sample.toLight = to_light;
        light_sample.irradiance = emittance;
        light_sample.pdf = pdf;
    } else {
        light_sample.toLight = vec3(0.f);
        light_sample.irradiance = vec3(0.f);
        light_sample.pdf = 0.f;
    }
}

// Output functions
#define ENABLE_RIS 1
#ifdef ENABLE_RIS
LightInfo sampleLights(inout Sampler rng, in Environment env, 
    in vec3 origin, in vec3 world_geo_normal,
    in vec3 world_shading_normal)
{
    uint32_t total_lights = env.numLights;
    float inv_selection_pdf = float(total_lights);
    
    const int num_ris_lights = 4;
    SceneAddresses scene_addrs = sceneAddrs[env.sceneID];

    vec3 selected_origin;
    vec3 selected_dir;
    float selected_dist;
    float selected_dist2;
    float selected_weight;
    float selected_cos_theta;
    uint32_t selected_mat = ~0u;
    float total_ris_weight = 0.f;

    for (int i = 0; i < num_ris_lights; i++) {
        TriangleLight light;
        {
            uint32_t light_idx = min(
                uint32_t(samplerGet1D(rng) * total_lights),
                total_lights - 1);

            PackedLight packed =
                lights[nonuniformEXT(env.baseLightOffset + light_idx)];
            light = unpackTriangleLight(scene_addrs.vertAddr,
                                        scene_addrs.idxAddr,
                                        packed.data);
        }

        vec2 light_sample_uv = samplerGet2D(rng);
        vec3 sampled_pos = getTriangleLightPoint(light, light_sample_uv);
        vec3 dir_check = sampled_pos - origin;

        vec3 shadow_offset_normal =
            dot(dir_check, world_geo_normal) > 0 ?
                world_geo_normal  : -world_geo_normal;

        vec3 shadow_origin =
            offsetRayOrigin(origin, shadow_offset_normal);

        vec3 to_light = sampled_pos - shadow_origin;
        
        float dist_to_light2 = dot(to_light, to_light);
        float dist_to_light = sqrt(dist_to_light2);
        to_light /= dist_to_light;

        vec3 c = cross(light.verts[1] - light.verts[0],
                       light.verts[2] - light.verts[0]);

        float c_len = length(c);
        vec3 tri_normal = c / c_len;
        float tri_area = 0.5f * c_len;

        float inv_source_pdf = tri_area * inv_selection_pdf;

        float cos_theta = abs(dot(tri_normal, to_light));
        float inv_dist2 = dist_to_light2 == 0.f ? 0.f : 1.f / dist_to_light2;
        float target_weight = cos_theta * inv_dist2;

        float ris_weight = target_weight * inv_source_pdf;

        float ris_selector = samplerGet1D(rng);

        total_ris_weight += ris_weight;

        // Hack to ensure first entry is selected
        // This shouldn't be necessary, but there is a weird situation
        // where ris_weight / total_ris_weight == 1 - epsilon,
        // on the first iteration, and ris_selector == 1 - epsilon
        if (selected_mat == ~0u ||
                ris_selector < (ris_weight / total_ris_weight)) {
            selected_origin = shadow_origin;
            selected_dir = to_light;
            selected_dist = dist_to_light;
            selected_dist2 = dist_to_light2;
            selected_weight = target_weight;
            selected_mat = light.matIdx;
            selected_cos_theta = cos_theta;
        }
    }

    // Normalize resampled weight and convert to solid angle PDF
    float pdf = total_ris_weight == 0 || selected_cos_theta == 0 ? 0.f :
        selected_weight * (num_ris_lights / total_ris_weight) *
            (selected_dist2 / selected_cos_theta);

    vec3 emittance = getMaterialEmittance(scene_addrs.matAddr,
                                          selected_mat);

    LightInfo info;
    info.lightSample.toLight = selected_dir;
    info.lightSample.irradiance = emittance;
    info.lightSample.pdf = pdf;
    info.shadowRayOrigin = selected_origin;

    // Hack, if shadow ray is right length, seem to be FP issues
    // where it still intersects
    info.shadowRayLength = selected_dist - 1e-6f;

    return info;
}

#else
LightInfo sampleLights(inout Sampler rng, in Environment env, 
    in vec3 origin, in vec3 base_normal, in vec3 shading_normal)
{
    //uint32_t total_lights = env.numLights + 1;
    uint32_t total_lights = env.numLights;
    //uint32_t total_lights = 1;

    uint32_t light_idx = uint32_t(samplerGet1D(rng) * total_lights);

    vec2 light_sample_uv = samplerGet2D(rng);

    float inv_selection_pdf = float(total_lights);

    SphereLight sphere_light = { vec3(0), 0.f, 0 };
    TriangleLight tri_light = {{ vec3(0), vec3(0), vec3(0) }, 0};
    PortalLight portal_light = {{ vec3(0), vec3(0), vec3(0), vec3(0) }};
    uint32_t light_type = 0;

    SceneAddresses scene_addrs = sceneAddrs[env.sceneID];

    if (light_idx < env.numLights) {
        light_type = unpackLight(env, scene_addrs.vertAddr, scene_addrs.idxAddr,
                                 light_idx, sphere_light, tri_light,
                                 portal_light);
    } else {
        light_type = LightTypeEnvironment;
    }

    vec3 light_position = vec3(0); 
    vec3 dir_check = vec3(0);
    LightSample light_sample = LightSample(vec3(0), vec3(0), 0.f);

    if (light_type == LightTypeSphere) {
        light_position = sphere_light.position;
        dir_check = light_position - origin;
    } else if (light_type == LightTypeTriangle) {
        light_position = getTriangleLightPoint(tri_light, light_sample_uv);
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
    if (light_type == LightTypeSphere) {
        sampleSphereLight(sphere_light,
                          scene_addrs.matAddr,
                          shadow_origin, inv_selection_pdf,
                          light_sample_uv, light_sample, shadow_len);
    } else if (light_type == LightTypeTriangle) {
        sampleTriangleLight(
            tri_light, scene_addrs.matAddr,
            shadow_origin, light_position, inv_selection_pdf,
            light_sample, shadow_len);
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

DeltaLightInfo getLight(in Environment env, int idx, vec3 origin)
{
    vec3 position = origin;
    vec3 irradiance = 0.3f * vec3(2.5f, 2.4f, 2.3f);
    if (idx == 0) {
        position = vec3(12.11, 1.469, -4.1517);
    } else if (idx == 1) {
        position = vec3(9.250230, 1.506868, -0.526671);
    } else if (idx == 2) {
        position = vec3(8.711510, 1.529880, -0.520881);
    } else if (idx == 3) {
        position = vec3(7.986629, 1.492613, -4.335118);
    } else if (idx == 4) {
        position = vec3(8.251885, 1.528721, -0.533173);
    }

    DeltaLightInfo info;
    info.position = position;
    info.irradiance = irradiance;

    return info;
}

#endif
