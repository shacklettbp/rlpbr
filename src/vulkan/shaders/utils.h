#ifndef RLPBR_VK_UTILS_H_INCLUDED
#define RLPBR_VK_UTILS_H_INCLUDED

vec3 octahedralVectorDecode(vec2 f) {
     f = f * 2.0 - 1.0;
     // https://twitter.com/Stubbesaurus/status/937994790553227264
     vec3 n = vec3(f.x, f.y, 1.f - abs(f.x) - abs(f.y));
     float t = clamp(-n.z, 0.0, 1.0);
     n.x += n.x >= 0.0 ? -t : t;
     n.y += n.y >= 0.0 ? -t : t;
     return normalize(n);
}

void decodeNormalTangent(in u32vec3 packed, out vec3 normal,
                         out vec4 tangentAndSign)
{
    vec2 ab = unpackHalf2x16(packed.x);
    vec2 cd = unpackHalf2x16(packed.y);

    normal = vec3(ab.x, ab.y, cd.x);
    float sign = cd.y;

    vec2 oct_tan = unpackSnorm2x16(packed.z);
    vec3 tangent = octahedralVectorDecode(oct_tan);

    tangentAndSign = vec4(tangent, sign);
}

vec3 transformPosition(mat4x3 o2w, vec3 p)
{
    return o2w[0] * p.x + o2w[1] * p.y + o2w[2] * p.z + o2w[3];
}

vec3 transformVector(mat4x3 o2w, vec3 v)
{
    return o2w[0] * v.x + o2w[1] * v.y + o2w[2] * v.z;
}

vec3 transformNormal(mat4x3 w2o, vec3 n)
{
    return vec3(dot(w2o[0], n), dot(w2o[1], n), dot(w2o[2], n));
}

#endif
