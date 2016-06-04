/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//------------------------------------------------------------------------

// Floats.

__device__ inline F32       fmaxf3          (F32 a, F32 b, F32 c)               { return fmaxf(fmaxf(a, b), c); }
__device__ inline F32       fminf3          (F32 a, F32 b, F32 c)               { return fminf(fminf(a, b), c); }
__device__ inline F32       smoothstep_n    (F32 f)                             { return (f*f)*(-2.f*f + 3.f); }
__device__ inline F32       smoothstep      (F32 f)                             { return (f < 0.f) ? 0.f : (f > 1.f) ? 1.f : smoothstep_n(f); }

// Vectors.

__device__ inline float3    operator*       (const float3& a, F32 b)            { return make_float3(a.x*b, a.y*b, a.z*b); }
__device__ inline float3    operator*       (F32 b, const float3& a)            { return make_float3(a.x*b, a.y*b, a.z*b); }
__device__ inline float4    operator*       (const float4& a, F32 b)            { return make_float4(a.x*b, a.y*b, a.z*b, a.w*b); }
__device__ inline float4    operator*       (F32 b, const float4& a)            { return make_float4(a.x*b, a.y*b, a.z*b, a.w*b); }
__device__ inline float3    operator+       (const float3& a, const float3& b)  { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ inline float4    operator+       (const float4& a, const float4& b)  { return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
__device__ inline float3    operator-       (const float3& a, const float3& b)  { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ inline float4    operator-       (const float4& a, const float4& b)  { return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
__device__ inline void      operator*=      (float3& a, F32 b)                  { a.x*=b; a.y*=b; a.z*=b; }
__device__ inline void      operator*=      (float4& a, F32 b)                  { a.x*=b; a.y*=b; a.z*=b; a.w*=b; }
__device__ inline void      operator/=      (float3& a, F32 b)                  { F32 ib = 1.f/b; a.x*=ib; a.y*=ib; a.z*=ib; }
__device__ inline void      operator/=      (float4& a, F32 b)                  { F32 ib = 1.f/b; a.x*=ib; a.y*=ib; a.z*=ib; a.w*=ib; }
__device__ inline void      operator+=      (float3& a, const float3& b)        { a.x+=b.x; a.y+=b.y; a.z+=b.z; }
__device__ inline void      operator+=      (float4& a, const float4& b)        { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; }
__device__ inline F32       dot             (const float3& a, const float3& b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline float3    normalize       (const float3& v)                   { F32 ilen = rsqrtf(dot(v, v)); return make_float3(v.x*ilen, v.y*ilen, v.z*ilen); }
__device__ inline float3    cross           (const float3& a, const float3& b)  { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ inline float3    scale           (const float3& a, const float3& b)  { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ inline float4    scale           (const float4& a, const float4& b)  { return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
__device__ inline F32       length          (const float3& v)                   { return sqrtf(dot(v, v)); }
__device__ inline float3    get             (volatile float3& v)                { return make_float3(v.x, v.y, v.z); }

// Matrices.

__device__ inline float4    operator*       (const Mat4f& m, const float4& v);
__device__ inline float3    operator*       (const Mat4f& m, const float3& v);
__device__ inline float3    operator*       (const Mat3f& m, const float3& v);
__device__ inline Mat3f     extractMat3f    (const Mat4f& m);

// Miscellaneous.

__device__ inline float4    fromABGR        (U32 abgr);
__device__ inline U32       toABGR          (float4 v);
__device__ inline void      jenkinsMix      (U32& a, U32& b, U32& c);
__device__ inline float3    perpendicular   (const float3& v);

//------------------------------------------------------------------------

__device__ inline float4 operator*(const Mat4f& m, const float4& v)
{
    return make_float4(
        m.m00 * v.x + m.m01 * v.y + m.m02 * v.z + m.m03 * v.w,
        m.m10 * v.x + m.m11 * v.y + m.m12 * v.z + m.m13 * v.w,
        m.m20 * v.x + m.m21 * v.y + m.m22 * v.z + m.m23 * v.w,
        m.m30 * v.x + m.m31 * v.y + m.m32 * v.z + m.m33 * v.w);
}

//------------------------------------------------------------------------

__device__ inline float3 operator*(const Mat4f& m, const float3& v)
{
    return make_float3(
        m.m00 * v.x + m.m01 * v.y + m.m02 * v.z + m.m03,
        m.m10 * v.x + m.m11 * v.y + m.m12 * v.z + m.m13,
        m.m20 * v.x + m.m21 * v.y + m.m22 * v.z + m.m23);
}

//------------------------------------------------------------------------

__device__ inline float3 operator*(const Mat3f& m, const float3& v)
{
    return make_float3(
        m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
        m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
        m.m20 * v.x + m.m21 * v.y + m.m22 * v.z);
}

//------------------------------------------------------------------------

__device__ inline Mat3f extractMat3f(const Mat4f& m)
{
    Mat3f n;
    n.m00 = m.m00; n.m01 = m.m01; n.m02 = m.m02;
    n.m10 = m.m10; n.m11 = m.m11; n.m12 = m.m12;
    n.m20 = m.m20; n.m21 = m.m21; n.m22 = m.m22;
    return n;
}

//------------------------------------------------------------------------

__device__ inline float4 fromABGR(U32 abgr)
{
    return make_float4(
        (F32)(abgr & 0xFF),
        (F32)((abgr >> 8) & 0xFF),
        (F32)((abgr >> 16) & 0xFF),
        (F32)(abgr >> 24));
}

//------------------------------------------------------------------------

__device__ inline U32 toABGR(float4 v)
{
    v.x = fminf(fmaxf(v.x, 0.f), 255.f);
    v.y = fminf(fmaxf(v.y, 0.f), 255.f);
    v.z = fminf(fmaxf(v.z, 0.f), 255.f);
    v.w = fminf(fmaxf(v.w, 0.f), 255.f);
    U32 ir = (U32)(v.x);
    U32 ig = (U32)(v.y);
    U32 ib = (U32)(v.z);
    U32 ia = (U32)(v.w);
    return ir | (ig << 8) | (ib << 16) | (ia << 24);
}

//------------------------------------------------------------------------

// By Bob Jenkins, 1996. bob_jenkins@burtleburtle.net.
__device__ inline void jenkinsMix(U32& a, U32& b, U32& c)
{
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);
}

//------------------------------------------------------------------------

__device__ inline float3 perpendicular(const float3& v)
{
    float vmin = fminf(fabsf(v.x), fminf(fabsf(v.y), fabsf(v.z)));
    if (vmin == fabsf(v.x))
        return make_float3(0, v.z, -v.y);
    else if (vmin == fabsf(v.y))
        return make_float3(-v.z, 0, v.x);
    else
        return make_float3(v.y, -v.x, 0);
}

//------------------------------------------------------------------------
