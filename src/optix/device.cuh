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

using ::min;
using ::max;
using ::fminf;
using ::fmaxf;
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

__device__ inline float2 make_float2(float s)
{
    return make_float2(s, s);
}

__device__ inline float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}

__device__ inline float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

__device__ inline float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

__device__ inline int2 make_int2(int s)
{
    return make_int2(s, s);
}

__device__ inline int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}

__device__ inline int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}

__device__ inline int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

__device__ inline uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}

__device__ inline uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}

__device__ inline uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

__device__ inline float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

__device__ inline float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}

__device__ inline float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}

__device__ inline float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}

__device__ inline float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

__device__ inline float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

__device__ inline int3 make_int3(int s)
{
    return make_int3(s, s, s);
}

__device__ inline int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}

__device__ inline int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}

__device__ inline int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

__device__ inline int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

__device__ inline uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}

__device__ inline uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}

__device__ inline uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}

__device__ inline uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}

__device__ inline uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

__device__ inline float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

__device__ inline float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}

__device__ inline float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}

__device__ inline float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

__device__ inline float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

__device__ inline int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}

__device__ inline int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}

__device__ inline int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}

__device__ inline int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

__device__ inline int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

__device__ inline uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}

__device__ inline uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}

__device__ inline uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}

__device__ inline uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

__device__ inline float2 make_float2(uint3 v)
{
    return make_float2(make_uint2(v));

}

__device__ inline float2 operator-(float2 &v)
{
    return make_float2(-v.x, -v.y);
}

__device__ inline float3 operator-(float3 &v)
{
    return make_float3(-v.x, -v.y, -v.z);
}

__device__ inline float4 operator-(float4 &v)
{
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}

__device__ inline int2 operator-(int2 &v)
{
    return make_int2(-v.x, -v.y);
}

__device__ inline int3 operator-(int3 &v)
{
    return make_int3(-v.x, -v.y, -v.z);
}

__device__ inline int4 operator-(int4 &v)
{
    return make_int4(-v.x, -v.y, -v.z, -v.w);
}

__device__ inline float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

__device__ inline float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}

__device__ inline float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}

__device__ inline void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

__device__ inline int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

__device__ inline void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}

__device__ inline int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}

__device__ inline int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}

__device__ inline void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

__device__ inline uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}

__device__ inline void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}

__device__ inline uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}

__device__ inline uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}

__device__ inline void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


__device__ inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ inline float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ inline void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

__device__ inline int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ inline int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}

__device__ inline void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

__device__ inline uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ inline uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}

__device__ inline void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

__device__ inline int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}

__device__ inline uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}

__device__ inline float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ inline float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

__device__ inline void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__device__ inline float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__device__ inline float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__device__ inline void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__device__ inline int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

__device__ inline void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__device__ inline int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

__device__ inline int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

__device__ inline void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__device__ inline uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

__device__ inline void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__device__ inline uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

__device__ inline uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

__device__ inline void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__device__ inline float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}

__device__ inline float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}

__device__ inline float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}

__device__ inline void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

__device__ inline int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}

__device__ inline void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}

__device__ inline int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}

__device__ inline int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}

__device__ inline void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

__device__ inline uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}

__device__ inline void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}

__device__ inline uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}

__device__ inline uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}

__device__ inline void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

__device__ inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__device__ inline float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__device__ inline float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}

__device__ inline void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__device__ inline int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__device__ inline int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}

__device__ inline int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}

__device__ inline void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__device__ inline uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__device__ inline uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}

__device__ inline uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}

__device__ inline void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__device__ inline float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__device__ inline void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

__device__ inline float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__device__ inline void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__device__ inline int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__device__ inline void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

__device__ inline int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__device__ inline int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}

__device__ inline void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__device__ inline uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__device__ inline void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

__device__ inline uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__device__ inline uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}

__device__ inline void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__device__ inline float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

__device__ inline void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}

__device__ inline float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

__device__ inline float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}

__device__ inline void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

__device__ inline int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}

__device__ inline void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}

__device__ inline int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}

__device__ inline int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}

__device__ inline void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

__device__ inline uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}

__device__ inline void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}

__device__ inline uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}

__device__ inline uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}

__device__ inline void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

__device__ inline float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__device__ inline float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

__device__ inline void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__device__ inline int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__device__ inline int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}

__device__ inline int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}

__device__ inline void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__device__ inline uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__device__ inline uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}

__device__ inline uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}

__device__ inline void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__device__ inline float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__device__ inline void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

__device__ inline float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__device__ inline float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__device__ inline void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__device__ inline int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__device__ inline void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

__device__ inline int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__device__ inline int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__device__ inline void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__device__ inline uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__device__ inline void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

__device__ inline uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__device__ inline uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__device__ inline void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__device__ inline float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

__device__ inline void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}

__device__ inline float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}

__device__ inline void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}

__device__ inline float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

__device__ inline float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ inline void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

__device__ inline float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

__device__ inline float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

__device__ inline float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}

__device__ inline void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

__device__ inline float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

__device__ inline void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

__device__ inline float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

__device__ inline float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

__device__ inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ inline int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
__device__ inline int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ inline uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
__device__ inline uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ inline float length(float2 v)
{
    return sqrtf(dot(v, v));
}

__device__ inline float length(float3 v)
{
    return sqrtf(dot(v, v));
}

__device__ inline float length(float4 v)
{
    return sqrtf(dot(v, v));
}

__device__ inline float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

__device__ inline float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

__device__ inline float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

__device__ inline float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ inline float lerp(float a, float b, float t)
{
    return fmaf(t, b, fmaf(-t, a, a));
}

__device__ inline float2 lerp(float2 a, float2 b, float t)
{
    return make_float2(
        lerp(a.x, b.x, t),
        lerp(a.y, b.y, t));
}

__device__ inline float3 lerp(float3 a, float3 b, float t)
{
    return make_float3(
        lerp(a.x, b.x, t),
        lerp(a.y, b.y, t),
        lerp(a.z, b.z, t));
}

__device__ inline float4 lerp(float4 a, float4 b, float t)
{
    return make_float4(
        lerp(a.x, b.x, t),
        lerp(a.y, b.y, t),
        lerp(a.z, b.z, t),
        lerp(a.w, b.w, t));
}

inline  __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

inline __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

inline  __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}

inline __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

inline __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}

inline __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

inline __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}

inline __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

inline __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}

inline __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

inline __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}

inline __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

inline __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

}
}
