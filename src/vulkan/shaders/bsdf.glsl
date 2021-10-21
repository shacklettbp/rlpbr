#ifndef RLPBR_VK_BSDF_GLSL_INCLUDED
#define RLPBR_VK_BSDF_GLSL_INCLUDED

// BSDFFlags "enum"
const uint32_t BSDFFlagsInvalid = 1 << 0;
const uint32_t BSDFFlagsDelta = 1 << 1;
const uint32_t BSDFFlagsDiffuse = 1 << 2;
const uint32_t BSDFFlagsMicrofacetReflection = 1 << 3;
const uint32_t BSDFFlagsMicrofacetTransmission = 1 << 4;

struct BSDFParams {
    vec3 rhoDiffuse;
    float transparencyMask;
    vec3 rhoTransmissive;
    float transmission;
    vec3 sharedF0;
    float sharedF90;
    vec3 transmissiveF0;
    float transmissiveF90;
    float alpha;
    float roughness;
    float diffuseLambertScale;
    float diffuseLookupF0;
    float diffuseAverageAlbedo;
    vec3 microfacetMSWeight;
    float microfacetMSAvgAlbedoComplement;

    // Sampling probabilities for each sub component of BSDF
    float diffuseProb;
    float microfacetProb;
    float microfacetMSProb;
    float transmissionProb;

#ifdef ADVANCED_MATERIAL
    float clearcoatScale;
    float clearcoatAlpha;
    float clearcoatProb;
#endif
};

struct SampleResult {
    vec3 dir;
    vec3 weight;
    uint32_t flags;
};

float fetchDiffuseAverageAlbedo(float lookup_f0, float roughness)
{
    return textureLod(sampler2D(msDiffuseAverageTexture, clampSampler),
        vec2(roughness, lookup_f0), 0.0).x;
}

float fetchDiffuseDirectionalAlbedo(float lookup_f0, float roughness,
                                    float cos_theta)
{
    return textureLod(sampler3D(msDiffuseDirectionalTexture, clampSampler),
        vec3(cos_theta, roughness, lookup_f0), 0.0).x;
}

float fetchMicrofacetMSAverageAlbedo(float roughness)
{
    return textureLod(sampler1D(msGGXAverageTexture, clampSampler),
        roughness, 0.0).x;
}

float fetchMicrofacetMSDirectionalAlbedo(float roughness, float cos_theta)
{
    return textureLod(sampler2D(msGGXDirectionalTexture, clampSampler),
        vec2(cos_theta, roughness), 0.0).x;
}

float sampleMSMicrofacetAngle(float roughness, float u)
{
    return textureLod(sampler2D(msGGXInverseTexture, clampSampler),
        vec2(u, roughness), 0.0).x;
}

float pdfMSMicrofacetAngle(float dir_albedo_compl, float avg_albedo_compl,
                           float wi_dot_n)
{
    return wi_dot_n * dir_albedo_compl / avg_albedo_compl;
}

float rgbToLuminance(vec3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

#define COMPUTE_FRESNEL(T)                                              \
    T computeFresnel(T f0, T f90, float cos_theta)                      \
    {                                                                   \
        float complement = max(1.f - cos_theta, 0.f);                   \
        return f0 + (f90 - f0) * complement * complement * complement * \
            complement * complement;                                    \
    }

COMPUTE_FRESNEL(float)
COMPUTE_FRESNEL(vec3)

#undef COMPUTE_FRESNEL

const float ShadingMinAlpha = 0.0064f;

BSDFParams buildBSDF(Material material, vec3 wo)
{
    const float prior_ior = 1.f;
    float ior_ratio = (material.ior - prior_ior) / (material.ior + prior_ior);
    float base_f0 = ior_ratio * ior_ratio;

    vec3 dielectric_f0 = min(vec3(1.f), base_f0 * 
        material.rhoSpecular) * material.specularScale;
    float dielectric_f90 = material.specularScale;

    // Core weights
    float transmission_weight = material.transmission;
    float opaque_weight = (1.f - material.transmission);
    float dielectric_weight = (1.f - material.metallic);

    vec3 base_dielectric = material.rho * dielectric_weight;

    // Microfacet params
    // Scale between specular and metallic fresnel
    vec3 shared_f0 = mix(material.rho, dielectric_f0, dielectric_weight);
    float shared_f90 = fma(dielectric_weight, dielectric_f90,
                           material.metallic);

    vec3 ss_fresnel_estimate =
        computeFresnel(shared_f0, vec3(shared_f90), wo.z);
    vec3 ss_transmissive_fresnel_estimate =
        1.f - computeFresnel(dielectric_f0, vec3(dielectric_f90), wo.z);

    float alpha = material.roughness * material.roughness;
    if (alpha < ShadingMinAlpha) {
        alpha = 0;
    }
    
    // Multiscattering / energy conservation params
    // FIXME, it is pretty ambiguous whether the lookup tables wants
    // shared_f0 or dielectric_f0 here
    float diffuse_lookup_f0 = max(shared_f0.x, max(shared_f0.y, shared_f0.z));

    float ms_microfacet_avg_albedo =
        fetchMicrofacetMSAverageAlbedo(material.roughness);
    vec3 ms_fresnel_avg = 1.f / 21.f * shared_f90 + 20.f / 21.f * shared_f0;

    float ms_avg_albedo_compl = 1.f - ms_microfacet_avg_albedo;
    float ms_dir_albedo_compl =
        (1.f - fetchMicrofacetMSDirectionalAlbedo(material.roughness, wo.z));

    vec3 ms_fresnel =
        (ms_fresnel_avg * ms_fresnel_avg * ms_microfacet_avg_albedo) /
        (1.f - ms_fresnel_avg * ms_avg_albedo_compl);

    vec3 ms_microfacet_weight = ms_fresnel * ms_dir_albedo_compl;

#ifdef ADVANCED_MATERIAL
    // Clearcoat params
    float clearcoat_reflectance_estimate =
        material.clearcoatScale * computeFresnel(0.04f, 1.f, wo.z);
    float clearcoat_alpha =
        material.clearcoatRoughness * material.clearcoatRoughness;
    if (clearcoat_alpha < ShadingMinAlpha) {
        clearcoat_alpha = 0;
    }

    float clearcoat_prob = clearcoat_reflectance_estimate;
    float not_clearcoat_prob = 1.f - clearcoat_prob;
#else
    const float not_clearcoat_prob = 1.f;
#endif

    // Compute importance sampling weights
    float dielectric_luminance =
        not_clearcoat_prob * rgbToLuminance(base_dielectric);
    float diffuse_prob =
        not_clearcoat_prob * dielectric_luminance * opaque_weight;
    float microfacet_prob =
        not_clearcoat_prob * rgbToLuminance(ss_fresnel_estimate);
    float microfacet_ms_prob =
        not_clearcoat_prob * rgbToLuminance(ms_fresnel);
    float transmission_prob = not_clearcoat_prob * dielectric_luminance *
        transmission_weight * rgbToLuminance(ss_transmissive_fresnel_estimate);

    float prob_sum = 
#ifdef ADVANCED_MATERIAL
        clearcoat_prob + 
#endif
        diffuse_prob + microfacet_prob +
        microfacet_ms_prob + transmission_prob;
    if (prob_sum > 0.f) {
        float inv_prob = 1.f / prob_sum;
        diffuse_prob *= inv_prob;
        microfacet_prob *= inv_prob;
        microfacet_ms_prob *= inv_prob;
        transmission_prob *= inv_prob;

#ifdef ADVANCED_MATERIAL
        clearcoat_prob *= inv_prob;
#endif
    }

    BSDFParams bsdf = {
        base_dielectric * opaque_weight,
        material.transparencyMask,
        base_dielectric * transmission_weight,
        transmission_weight,
        shared_f0,
        shared_f90,
        dielectric_f0,
        dielectric_f90,
        alpha,
        material.roughness,
        material.specularScale,
        diffuse_lookup_f0,
        fetchDiffuseAverageAlbedo(diffuse_lookup_f0, material.roughness),
        ms_microfacet_weight,
        ms_avg_albedo_compl,
        diffuse_prob,
        microfacet_prob,
        microfacet_ms_prob,
        transmission_prob,
#ifdef ADVANCED_MATERIAL
        material.clearcoatScale,
        clearcoat_alpha,
        clearcoat_prob,
#endif
    };

    return bsdf;
}

// Enterprise PBR diffuse BRDF
// diffuseWeight and diffuseBSDF are separated to allow concentricHemisphere
// sampling which divides out the M_1_PI * wi_dot_n factor when sampling
vec3 diffuseWeight(BSDFParams params, float wo_dot_n, float wi_dot_n,
                   out bool invalid)
{
    float E_diffuse_o = fetchDiffuseDirectionalAlbedo(params.diffuseLookupF0,
                                                      params.roughness,
                                                      wo_dot_n);

    float E_diffuse_i = fetchDiffuseDirectionalAlbedo(params.diffuseLookupF0,
                                                      params.roughness,
                                                      wi_dot_n);

    float Bc = ((1.f - E_diffuse_o) * (1.f - E_diffuse_i)) /
            (1.f - params.diffuseAverageAlbedo);

    float weight = mix(1.f, Bc, params.diffuseLambertScale);

    if (min(wo_dot_n, wi_dot_n) < NEAR_ZERO) {
        invalid = true;
        return vec3(0.f);
    } else {
        invalid = false;
        return weight * params.rhoDiffuse;
    }
}

vec3 diffuseBSDF(BSDFParams bsdf_params, float wo_dot_n, float wi_dot_n,
                 out float pdf)
{
    bool invalid;
    vec3 weight = diffuseWeight(bsdf_params, wo_dot_n, wi_dot_n, invalid);
    pdf = M_1_PI * wi_dot_n;

    if (invalid) {
        pdf = 0.f;
    }

    return weight * pdf;
}

SampleResult sampleDiffuse(BSDFParams bsdf_params, vec3 wo, vec2 sample_uv)
{
    vec3 wi = concentricHemisphere(sample_uv);
    bool invalid;
    vec3 weight = diffuseWeight(bsdf_params, wo.z, wi.z, invalid);

    SampleResult result = {
        wi,
        weight,
        BSDFFlagsDiffuse,
    };

    return result;
}

// Single scattering GGX Microfacet BRDF
float ggxLambda(float cos_theta, float a2)
{
    float cos2 = cos_theta * cos_theta;
    float tan2 = max(1.f - cos2, 0.f) / cos2;
    float l = 0.5f * (-1.f + sqrt(1.f + a2 * tan2));

    return cos_theta <= 0.f ? 0.f : l;
}

float ggxG1(float cos_theta, float a2)
{
    float cos2 = cos_theta * cos_theta;
    float tan2 = max(1.f - cos2, 0.f) / cos2;
    float g1 = 2.f / (1.f + sqrt(1.f + a2 * tan2));
    return cos_theta <= 0.f ? 0.f : g1;
}

float ggxNDF(float alpha, float cos_theta)
{
    float a2 = alpha * alpha;
    float d = ((cos_theta * a2 - cos_theta) * cos_theta + 1.f);
    return a2 / (d * d * M_PI);
}

float ggxMasking(float a2, float out_cos, float in_cos)
{
    float in_lambda = ggxLambda(in_cos, a2);
    float out_lambda = ggxLambda(out_cos, a2);
    return 1.f / (1.f + in_lambda + out_lambda);
}

#define EVAL_GGX(T)                                                       \
    T evalGGX(float wo_dot_n, float wi_dot_n, float n_dot_h, T F,         \
              float alpha, out float pdf)                                 \
    {                                                                     \
        float a2 = alpha * alpha;                                         \
        float D = ggxNDF(alpha, n_dot_h);                                 \
        float G = ggxMasking(a2, wo_dot_n, wi_dot_n);                     \
                                                                          \
        float common_weight = 0.25f * D / wo_dot_n;                       \
        T specular = common_weight * F * G;                               \
        pdf = common_weight * ggxG1(wo_dot_n, a2);                        \
                                                                          \
        if (alpha == 0.f || min(wo_dot_n, wi_dot_n) < NEAR_ZERO) {        \
            specular = T(0.0);                                            \
            pdf = 0.f;                                                    \
        }                                                                 \
                                                                          \
        return specular;                                                  \
    }

EVAL_GGX(float)
EVAL_GGX(vec3)

#undef EVAL_GGX

vec3 microfacetBSDF(BSDFParams params, float wo_dot_n, float wi_dot_n,
                    float n_dot_h, float dir_dot_h, out float pdf)
{
    vec3 F = computeFresnel(params.sharedF0, vec3(params.sharedF90),
                            dir_dot_h);

    return evalGGX(wo_dot_n, wi_dot_n, n_dot_h, F, params.alpha, pdf);
}

vec3 microfacetTransmissiveBSDF(BSDFParams params, vec3 wo, vec3 wi,
                                out float pdf)
{
    wi.z *= -1.f;

    float wi_dot_wo = dot(wo, wi);

    float len_sq_io = 2.f + 2.f * wi_dot_wo;
    float rlen_io = inversesqrt(len_sq_io);

    float n_dot_h = (wo.z + wi.z) * rlen_io;
    float dir_dot_h = rlen_io + rlen_io * wi_dot_wo;

    vec3 F = 1.f - computeFresnel(params.transmissiveF0,
                                  vec3(params.transmissiveF90),
                                  dir_dot_h);

    vec3 microfacet_response = evalGGX(wo.z, wi.z,
        n_dot_h, F, params.alpha, pdf);

    return microfacet_response * params.rhoTransmissive;
}

vec3 sampleGGX(float alpha, vec3 wo, vec2 sample_uv)
{
    vec3 Vh = normalize(vec3(alpha * wo.x, alpha * wo.y, wo.z));

    // Construct orthonormal basis (Vh,T1,T2).
    vec3 T1 = (Vh.z < 0.9999f) ?
        normalize(cross(vec3(0.f, 0.f, 1.f), Vh)) :
        vec3(1.f, 0.f, 0.f);

    vec3 T2 = cross(Vh, T1);

    float r = sqrt(sample_uv.x);
    float phi = (2.f * M_PI) * sample_uv.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;
    vec3 h = normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.f, Nh.z)));

    return h;
}

#define SAMPLE_MICROFACET(T)                                                \
    void sampleMicrofacet(in vec3 wo, in vec2 sample_uv, in T F0,           \
                          in T F90, in float alpha, bool transmission,      \
                          out vec3 wi, out T weight, out uint32_t flags)    \
    {                                                                       \
        wi = vec3(0);                                                       \
        weight = T(0);                                                      \
        flags = 0;                                                          \
        if (wo.z < NEAR_ZERO) {                                             \
            wi = vec3(0);                                                   \
            weight = T(0);                                                  \
                                                                            \
            flags |= BSDFFlagsInvalid;                                      \
        } else if (alpha == 0.f) {                                          \
            wi = vec3(-wo.x, -wo.y, wo.z);                                  \
            T F = computeFresnel(F0, F90, wo.z);                            \
            if (transmission) {                                             \
                weight = T(1.f) - F;                                        \
            } else {                                                        \
                weight = F;                                                 \
            }                                                               \
                                                                            \
            flags |= BSDFFlagsDelta;                                        \
        } else {                                                            \
            vec3 h = sampleGGX(alpha, wo, sample_uv);                       \
            float cos_half_out = dot(wo, h);                                \
            wi = 2.f * cos_half_out * h - wo;                               \
                                                                            \
            float a2 = alpha * alpha;                                       \
                                                                            \
            float G = ggxMasking(a2, wo.z, wi.z);                           \
            float GG1_out = G * (1.f + ggxLambda(wo.z, a2));                \
            T F = computeFresnel(F0, F90, cos_half_out);                    \
                                                                            \
            if (wi.z < NEAR_ZERO) {                                         \
                weight = T(0);                                              \
            } else {                                                        \
                if (transmission) {                                         \
                    weight = (T(1) - F) * GG1_out;                          \
                } else {                                                    \
                    weight = F * GG1_out;                                   \
                }                                                           \
            }                                                               \
                                                                            \
            if (transmission) {                                             \
                flags |= BSDFFlagsMicrofacetTransmission;                   \
            } else {                                                        \
                flags |= BSDFFlagsMicrofacetReflection;                     \
            }                                                               \
        }                                                                   \
    }

SAMPLE_MICROFACET(float)
SAMPLE_MICROFACET(vec3)

#undef SAMPLE_MICROFACET

SampleResult sampleMicrofacetShared(
    BSDFParams params, vec3 wo, vec2 sample_uv)
{
    vec3 wi;
    vec3 weight;
    uint32_t flags;
    sampleMicrofacet(wo, sample_uv, params.sharedF0, vec3(params.sharedF90),
                     params.alpha, false, wi, weight, flags);

    SampleResult result = {
        wi,
        weight,
        flags,
    };

    return result;
}

SampleResult sampleMicrofacetTransmission(
    BSDFParams params, vec3 wo, vec2 sample_uv)
{
    vec3 wi;
    vec3 weight;
    uint32_t flags;
    sampleMicrofacet(wo, sample_uv, params.transmissiveF0,
                     vec3(params.transmissiveF90),
                     params.alpha, true, wi, weight, flags);
    

    // Reflect back for transmission
    wi.z *= -1.f;
    weight *= params.rhoTransmissive;

    SampleResult result = {
        wi,
        weight,
        flags,
    };

    return result;
}

vec3 microfacetMSBSDF(BSDFParams params, float wo_dot_n, float wi_dot_n,
                      out float pdf)
{
    float common_weight = wi_dot_n;

    float dir_albedo_compl =
        1.f - fetchMicrofacetMSDirectionalAlbedo(params.roughness, wi_dot_n);

    pdf = M_1_PI * pdfMSMicrofacetAngle(dir_albedo_compl,
                               params.microfacetMSAvgAlbedoComplement,
                               wi_dot_n);

    vec3 ms_contrib = wi_dot_n * M_1_PI * dir_albedo_compl *
        params.microfacetMSWeight /
        params.microfacetMSAvgAlbedoComplement;

    if (params.alpha == 0.f || min(wo_dot_n, wi_dot_n) < NEAR_ZERO) {
        ms_contrib = vec3(0.f);
        pdf = 0.f;
    }

    return ms_contrib;
}

SampleResult sampleMSMicrofacet(BSDFParams params, vec3 wo, vec2 sample_uv)
{
    float theta = sampleMSMicrofacetAngle(params.roughness, sample_uv.x);
    float phi = 2.0 * M_PI * sample_uv.y;
    vec2 circle_dir = vec2(cos(phi), sin(phi));
    float xy_mag = sqrt(1.0 - theta * theta);

    vec3 wi = vec3(circle_dir.x * xy_mag,
                   circle_dir.y * xy_mag, theta);

    // 2pi in pdf cancels out
    vec3 weight = params.microfacetMSWeight;

    SampleResult result = {
        wi,
        weight,
        BSDFFlagsMicrofacetReflection,
    };

    if (min(wo.z, wi.z) < NEAR_ZERO) {
        result.dir = vec3(0.f);
        result.weight = vec3(0.f);
        result.flags = BSDFFlagsInvalid;
    }

    return result;
}

#ifdef ADVANCED_MATERIAL
void clearcoatBSDF(in BSDFParams params, in float wo_dot_n, in float wi_dot_n,
                   in float n_dot_h, in float dir_dot_h,
                   out float clearcoat_response, out float base_scale,
                   out float pdf)
{
    float F = computeFresnel(0.04f, 1.f, dir_dot_h);

    float response = evalGGX(wo_dot_n, wi_dot_n, n_dot_h, F,
                             params.clearcoatAlpha, pdf);

    float max_fresnel_n = max(computeFresnel(0.04f, 1.f, wo_dot_n),
                              computeFresnel(0.04f, 1.f, wi_dot_n));

    clearcoat_response = response * params.clearcoatScale;
    base_scale = 1.f - params.clearcoatScale * max_fresnel_n;
}

SampleResult sampleClearcoat(BSDFParams params, vec3 wo, vec2 sample_uv)
{
    vec3 wi;
    float weight;
    uint32_t flags;
    sampleMicrofacet(wo, sample_uv, 0.04f, 1.f, params.clearcoatAlpha, false,
                     wi, weight, flags);
    weight *= params.clearcoatScale;

    SampleResult result = {
        wi,
        vec3(weight),
        flags,
    };
}
#endif

float diffusePDF(BSDFParams params, float wi_dot_n)
{
    if (wi_dot_n < NEAR_ZERO) {
        return 0.f;
    } else {
        return M_1_PI * wi_dot_n;
    }
}

float microfacetPDF(BSDFParams params, float wo_dot_n, float wi_dot_n,
                    float n_dot_h, float dir_dot_h)
{
    float a2 = params.alpha * params.alpha;
    float D = ggxNDF(a2, n_dot_h);
    float G1 = ggxG1(wo_dot_n, a2);

    float pdf = 0.25f * D * G1 / wo_dot_n;

    if (params.alpha == 0.f || min(wo_dot_n, wi_dot_n) < NEAR_ZERO) {
        pdf = 0.f;
    }

    return pdf;
}

float microfacetMSPDF(BSDFParams params, float wi_dot_n)
{
    float dir_albedo = fetchMicrofacetMSDirectionalAlbedo(params.roughness,
                                                          wi_dot_n);

    float theta_pdf = pdfMSMicrofacetAngle(1.f - dir_albedo,
        params.microfacetMSAvgAlbedoComplement,
        wi_dot_n);

    float pdf = theta_pdf * M_1_PI;

    if (wi_dot_n < NEAR_ZERO) {
        pdf = 0.f;
    }

    return pdf;
}

float microfacetTransmissivePDF(BSDFParams params, vec3 wo, vec3 wi)
{
    wi.z *= -1.f;

    float wi_dot_wo = dot(wo, wi);

    float len_sq_io = 2.f + 2.f * wi_dot_wo;
    float rlen_io = inversesqrt(len_sq_io);

    float n_dot_h = (wo.z + wi.z) * rlen_io;
    float dir_dot_h = rlen_io + rlen_io * wi_dot_wo;

    return microfacetPDF(params, wo.z, wi.z, n_dot_h, dir_dot_h);
}

vec3 evalBSDF(BSDFParams params, vec3 wo, vec3 wi, out float pdf)
{
    // Hammon 2017
    float wi_dot_wo = dot(wo, wi);

    float len_sq_io = 2.f + 2.f * wi_dot_wo;
    float rlen_io = inversesqrt(len_sq_io);

    float n_dot_h = (wo.z + wi.z) * rlen_io;
    float dir_dot_h = rlen_io + rlen_io * wi_dot_wo;

    float diffuse_pdf;
    vec3 diffuse = diffuseBSDF(params, wo.z, wi.z, diffuse_pdf);

    float microfacet_pdf;
    vec3 microfacet =
        microfacetBSDF(params, wo.z, wi.z, n_dot_h, dir_dot_h, microfacet_pdf);

    float microfacet_ms_pdf;
    vec3 microfacet_ms =
        microfacetMSBSDF(params, wo.z, wi.z, microfacet_ms_pdf);

    float transmissive_pdf;
    vec3 transmissive =
        microfacetTransmissiveBSDF(params, wo, wi, transmissive_pdf);

    vec3 base = diffuse + microfacet + microfacet_ms + transmissive;
    float base_pdf = params.diffuseProb * diffuse_pdf +
                     params.microfacetProb * microfacet_pdf +
                     params.microfacetMSProb * microfacet_ms_pdf +
                     params.transmissionProb * transmissive_pdf;
#ifdef ADVANCED_MATERIAL
    float clearcoat_response, base_scale, clearcoat_pdf;
    clearcoatBSDF(params, wo.z, wi.z, n_dot_h, dir_dot_h, clearcoat_response,
                  base_scale, clearcoat_pdf);

    pdf = base_pdf + params.clearcoatProb * clearcoat_pdf;
    return base * base_scale + clearcoat_response;
#else
    pdf = base_pdf;
    return base;
#endif
}

float pdfBSDF(BSDFParams params, vec3 wo, vec3 wi)
{
    // Hammon 2017
    float wi_dot_wo = dot(wo, wi);

    float len_sq_io = 2.f + 2.f * wi_dot_wo;
    float rlen_io = inversesqrt(len_sq_io);

    float n_dot_h = (wo.z + wi.z) * rlen_io;
    float dir_dot_h = rlen_io + rlen_io * wi_dot_wo;

    float diffuse = diffusePDF(params, wi.z);

    float microfacet = microfacetPDF(params, wo.z, wi.z, n_dot_h, dir_dot_h);

    float microfacet_ms = microfacetMSPDF(params, wi.z);

    float transmissive = microfacetTransmissivePDF(params, wo, wi);

    float pdf = params.diffuseProb * diffuse +
        params.microfacetProb * microfacet + 
        params.microfacetMSProb * microfacet_ms +
        params.transmissionProb * transmissive;

#ifdef ADVANCED_MATERIAL
    float clearcoat_pdf = clearcoatPDF(params, wo.z, wi.z, n_dot_h, dir_dot_h);

    return params.clearcoatProb * clearcoat_pdf + pdf;
#else
    return pdf;
#endif
}

SampleResult sampleBSDF(inout Sampler rng,
                        in BSDFParams params,
                        in vec3 wo)
{
    float selector = samplerGet1D(rng);
    vec2 uv = samplerGet2D(rng);

    float cdf[]  = {
        params.diffuseProb,
        params.microfacetProb,
        params.microfacetMSProb,
        params.transmissionProb,
#ifdef ADVANCED_MATERIAL
        params.clearcoatProb,
#endif
    };

    cdf[1] += cdf[0];
    cdf[2] += cdf[1];
    cdf[3] += cdf[2];

#ifdef ADVANCED_MATERIAL
    cdf[4] += cdf[3];
#endif

    SampleResult result = {
        vec3(0),
        vec3(0),
        BSDFFlagsInvalid,
    };

    if (selector < cdf[0]) {
        result = sampleDiffuse(params, wo, uv);
        result.weight /= params.diffuseProb;
    } else if (selector < cdf[1]) {
        result = sampleMicrofacetShared(params, wo, uv);
        result.weight /= params.microfacetProb;
    } else if (selector < cdf[2]) {
        result = sampleMSMicrofacet(params, wo, uv);
        result.weight /= params.microfacetMSProb;
    } else if (selector < cdf[3]) {
        result = sampleMicrofacetTransmission(params, wo, uv);
        result.weight /= params.transmissionProb;
    } 
#ifdef ADVANCED_MATERIAL
    else if (selector < cdf[4]) {
        result = sampleClearcoat(params, wo, uv);
        result.weight /= params.clearcoatProb;
    }
#endif

    return result;
}

#endif
