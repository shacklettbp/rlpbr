#ifndef RLPBR_VK_RESERVOIR_GLSL_INCLUDED
#define RLPBR_VK_RESERVOIR_GLSL_INCLUDED

#include "shader_common.h"
#include "sampler.glsl"
#include "math.glsl"
#include "bsdf.glsl"

#define RESERVOIR_M (512)

vec3 reservoirSampleP(in BSDFParams bsdf_params, inout Sampler rng,
                      out float pdf)
{
    // FIXME
    return vec3(0.f, 0.f, 1.f);
}

// Bitterli algo 2. Note that this does not set r.W or r.M
void updateReservoir(inout Reservoir r, in vec3 xi, in float wi,
                     in float sample_pdf,
                     inout Sampler rng)
{
    r.wSum += wi;

    if (samplerGet1D(rng) < wi / r.wSum) {
        r.y = xi;
        r.pHat = wi * sample_pdf;
    }
}

// Bitterli equation 6
void reservoirUpdateW(inout Reservoir r)
{
    r.W = (1.f / r.pHat) * ((1.f / float(r.M)) * r.wSum);
}

// Bitterli algo 3.
// p_hat is bsdf_pdf * light_pdf
// p is light_pdf - therefore p_hat / p = bsdf_pdf
Reservoir initReservoirRIS(in BSDFParams bsdf_params, in vec3 wo,
                           in int M, inout Sampler rng)
{
    Reservoir r;
    float init_sample_pdf;
    r.y = reservoirSampleP(bsdf_params, rng, init_sample_pdf);
    r.wSum = pdfBSDF(bsdf_params, wo, r.y);
    r.pHat = r.wSum * init_sample_pdf;

    for (int i = 1; i < M; i++) {
        float sample_pdf;
        vec3 xi = reservoirSampleP(bsdf_params, rng, sample_pdf);
        updateReservoir(r, xi, pdfBSDF(bsdf_params, wo, xi), sample_pdf,
                        rng);
    }

    r.M = M;
    reservoirUpdateW(r);
    r.pad = 0;

    return r;
}

// Subset of Bitterli algo 4, merge b into a
void mergeReservoir(inout Reservoir a, in Reservoir b, in vec3 outgoing,
                    in BSDFParams bsdf_params, inout Sampler rng)
{
    updateReservoir(a, b.y, b.pHat * b.W * b.M, 1.f, rng);

    a.M += b.M;
    reservoirUpdateW(a);
}

#endif
