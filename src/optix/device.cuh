#pragma once

#include <cuda/std/cstdint>

namespace RLpbr {
namespace optix {

using ulong = unsigned long;
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;

using cuda::std::int8_t;
using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int64_t;
using cuda::std::uint8_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

using ::make_float2;
using ::make_float3;
using ::make_float4;
using ::make_int2;
using ::make_int3;
using ::make_int4;
using ::make_uint2;
using ::make_uint3;
using ::make_uint4;

// Subset of cuda sample's helper_math.h + some additions

inline float2 make_float2(float s)
{
    return make_float2(s, s);
}

inline float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}

inline float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline int2 make_int2(int s)
{
    return make_int2(s, s);
}

inline int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}

inline int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}

inline uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}

inline uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

inline float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}

inline float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}

inline float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}

inline float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline int3 make_int3(int s)
{
    return make_int3(s, s, s);
}

inline int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}

inline int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}

inline int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}

inline uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}

inline uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}

inline uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}

inline uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

inline float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}

inline float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}

inline float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}

inline int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}

inline int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}

inline int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

inline int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

inline uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}

inline uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}

inline uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}

inline uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

inline float2 make_float2(uint3 v)
{
    return make_float2(make_uint2(v));

}

inline float2 operator-(float2 &v)
{
    return make_float2(-v.x, -v.y);
}

inline float3 operator-(float3 &v)
{
    return make_float3(-v.x, -v.y, -v.z);
}

inline float4 operator-(float4 &v)
{
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}

inline int2 operator-(int2 &v)
{
    return make_int2(-v.x, -v.y);
}

inline int3 operator-(int3 &v)
{
    return make_int3(-v.x, -v.y, -v.z);
}

inline int4 operator-(int4 &v)
{
    return make_int4(-v.x, -v.y, -v.z, -v.w);
}

inline float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

inline void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}

inline float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}

inline void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

inline void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}

inline int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}

inline void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}

inline void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}

inline uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}

inline void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}

inline void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}

inline void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}

inline uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}

inline float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

inline int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

inline void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

inline uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

inline void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}

inline float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}

inline float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}

inline void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}

inline void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}

inline int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}

inline int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}

inline void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}

inline void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}

inline uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}

inline uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}

inline void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

inline float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}

inline void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}

inline int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}

inline void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}

inline uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}

inline void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

inline void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

inline int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}

inline void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

inline uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}

inline void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

inline void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}

inline float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

inline float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}

inline void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}

inline void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}

inline int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}

inline int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}

inline void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}

inline void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}

inline uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}

inline uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}

inline void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}

inline int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}

inline uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}

inline void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

inline void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}

inline float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}

inline void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}

inline float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

inline float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}

inline void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

inline float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

inline void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

inline float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

inline float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline float length(float2 v)
{
    return sqrtf(dot(v, v));
}

inline float length(float3 v)
{
    return sqrtf(dot(v, v));
}

inline float length(float4 v)
{
    return sqrtf(dot(v, v));
}

inline float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

inline float2 lerp(float2 a, float2 b, float t)
{
    return a + t * (b - a);
}

inline float3 lerp(float3 a, float3 b, float t)
{
    return a + t * (b - a);
}

inline float4 lerp(float4 a, float4 b, float t)
{
    return a + t * (b - a);
}

}
}
