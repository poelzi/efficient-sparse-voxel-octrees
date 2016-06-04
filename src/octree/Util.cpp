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

#include "Util.hpp"

using namespace FW;

//------------------------------------------------------------------------

namespace FW
{

static void     decodeDXTColorHead  (Vec3f* ref, U32 head);
static U32      encodeDXTNormalAxis (U32 headUV, const Vec3f& axis, int shift);
static Vec3f    decodeDXTNormalAxis (U32 headUV, int shift);

template <class S> __forceinline void          findMinMax             (S x0, S x1, S x2, S& min, S& max);
template <class S, class V> __forceinline bool planeBoxOverlap        (const V& normal, S d, const V& maxbox);
template <class S, class V> __forceinline bool isectsDeltaTriangleBox (const V& p, const V& pu, const V& pv, const V& boxHalfSize);

template <class S, class V2, class V3> __forceinline int clipDeltaTriangleToBox (V2 baryOut[9], const V3& p, const V3& pu, const V3& pv, const V3& boxHalfSize);

}

//------------------------------------------------------------------------

Vec3i FW::getCubeChildPos(const Vec3i& parentPos, int parentScale, int childIdx)
{
    FW_ASSERT(parentScale > 0);
    FW_ASSERT(childIdx >= 0 && childIdx < 8);

    Vec3i childPos = parentPos;
    int childSize = 1 << (parentScale - 1);

    if ((childIdx & 1) != 0) childPos.x += childSize;
    if ((childIdx & 2) != 0) childPos.y += childSize;
    if ((childIdx & 4) != 0) childPos.z += childSize;

    return childPos;
}

//------------------------------------------------------------------------

int FW::getCubeChildIndex(const Vec3i& childPos, int parentScale)
{
    FW_ASSERT(parentScale > 0);
    int shift = parentScale - 1;
    return
        ((childPos.x >> shift) & 1) |
        (((childPos.y >> shift) & 1) << 1) |
        (((childPos.z >> shift) & 1) << 2);
}

//------------------------------------------------------------------------

bool FW::isPosInCube(const Vec3i& pos, const Vec3i& cubePos, int cubeScale)
{
    Vec3i rel = pos - cubePos;
    return (rel.min() >= 0 && rel.max() < (1 << cubeScale));
}

//------------------------------------------------------------------------

U32 FW::encodeRawNormal(const Vec3f& normal)
{
    Vec3f a(abs(normal.x), abs(normal.y), abs(normal.z));
    int axis = (a.x >= max(a.y, a.z)) ? 0 : (a.y >= a.z) ? 1 : 2;

    Vec3f tuv;
    switch (axis)
    {
    case 0:  tuv = normal; break;
    case 1:  tuv = Vec3f(normal.y, normal.z, normal.x); break;
    default: tuv = Vec3f(normal.z, normal.x, normal.y); break;
    }

    return
        ((tuv.x >= 0.0f) ? 0 : 0x80000000) |
        (axis << 29) |
        ((clamp((S32)((tuv.y / fabsf(tuv.x)) * 16383.0f), -0x4000, 0x3FFF) & 0x7FFF) << 14) |
        (clamp((S32)((tuv.z / fabsf(tuv.x)) * 8191.0f),   -0x2000, 0x1FFF) & 0x3FFF);
}

//------------------------------------------------------------------------

Vec3i FW::decodeRawNormal(U32 value)
{
    S32 sign = (S32)value >> 31;
    S32 axis = (value >> 29) & 3;
    S32 t = (sign ^ 0x7fffffff);
    S32 u = (value << 3);
    S32 v = (value << 18);

    switch (axis)
    {
    case 0:  return Vec3i(t, u, v);
    case 1:  return Vec3i(v, t, u);
    default: return Vec3i(u, v, t);
    }
}

//------------------------------------------------------------------------

S32 FW::encodeContourNormal(const Vec3f& normal)
{
    Vec3f a = normal.abs();
    int i = (a.x >= a.y && a.x >= a.z) ? 0 : (a.y >= a.x && a.y >= a.z) ? 1 : 2;
    Vec3f n = normal * (31.0f / max(a[i], 1.0e-16f));
    n[i] = (n[i] >= 0.0f) ? 31.0f : -31.0f;

    S32 value;
    value  = clamp((S32)(n.z + 32.5f), 0, 63) ^ 32;
    value |= (clamp((S32)(n.y - (F32)value * exp2(-6) + 32.5f), 0, 63) ^ 32) << 6;
    value |= (clamp((S32)(n.x - (F32)value * exp2(-12) + 32.5f), 0, 63) ^ 32) << 12;
    return value;
}

//------------------------------------------------------------------------

S32 FW::encodeContourBounds(S32 value, F32 lo, F32 hi)
{
    value |= (clamp((S32)((lo + hi) * (2.0f / 3.0f) - (F32)value * exp2(-18) + 64.5f), 0, 127) ^ 64) << 18;
    F32 pos = (F32)(value << 7) * exp2(-25) * (3.0f / 4.0f);
    F32 thick = fastMax(pos - lo, hi - pos) * 2.0f;
    value |= clamp((S32)(thick * (4.0f / 3.0f) - (F32)value * exp2(-25) + (1.0f - 1.0e-5f)), 0, 127) << 25;
    return value;
}

//------------------------------------------------------------------------

Vec3f FW::decodeContourNormal(S32 value)
{
    return Vec3f(
        (F32)(value << 14) * exp2(-26),
        (F32)(value << 20) * exp2(-26),
        (F32)(value << 26) * exp2(-26));
}

//------------------------------------------------------------------------

Vec2f FW::decodeContourPosThick(S32 value)
{
    return Vec2f(
        (F32)(value << 7) * exp2(-25) * (3.0f / 4.0f),
        (F32)(U32)value * (3.0f / 4.0f) * exp2(-25));
}

//------------------------------------------------------------------------

S32 FW::xformContourToChild(S32 value, int childIdx)
{
    Vec3f normal = decodeContourNormal(value);
    Vec2f posThick = decodeContourPosThick(value);

    F32 pos = posThick.x * 2.0f;
    pos += normal.x * (0.5f - (F32)(childIdx & 1));
    pos += normal.y * (0.5f - (F32)((childIdx >> 1) & 1));
    pos += normal.z * (0.5f - (F32)(childIdx >> 2));

    return encodeContourBounds(
        value & 0x0003FFFF,
        fastClamp(pos - posThick.y, -48.0f, 48.0f),
        fastClamp(pos + posThick.y, -48.0f, 48.0f));
}

//------------------------------------------------------------------------

bool FW::isectPolyWithContour(ConvexPolyhedron& poly, S32 contour, int planeID)
{
    Vec3f normal = decodeContourNormal(contour);
    Vec2f posThick = decodeContourPosThick(contour);
    bool a = poly.intersect(Vec4f(normal, -posThick.x - posThick.y * 0.5f), planeID);
    bool b = poly.intersect(Vec4f(-normal, posThick.x - posThick.y * 0.5f), planeID);
    return (a || b);
}

//------------------------------------------------------------------------

void FW::decodeDXTColorHead(Vec3f* ref, U32 head)
{
    ref[0] = Vec3f((F32)(head << 16), (F32)(head << 21), (F32)(head << 27)) * exp2(-32);
    ref[1] = Vec3f((F32)head, (F32)(head << 5), (F32)(head << 11)) * exp2(-32);
    ref[2] = ref[0] * (2.0f / 3.0f) + ref[1] * (1.0f / 3.0f);
    ref[3] = ref[0] * (1.0f / 3.0f) + ref[1] * (2.0f / 3.0f);
}

//------------------------------------------------------------------------

U64 FW::encodeDXTColors(const Vec3f* colors, const S32* indices, int num)
{
    // Use the two colors that are furthest away as references.

    FW_ASSERT(num > 0);
    Vec2i refIdx;
    F32 refDist = -FW_F32_MAX;
    for (int i = 0; i < num; i++)
    {
        for (int j = i + 1; j < num; j++)
        {
            F32 tmpDist = (colors[i] - colors[j]).lenSqr();
            if (tmpDist > refDist)
            {
                refIdx = Vec2i(i, j);
                refDist = tmpDist;
            }
        }
    }

    // Encode the reference colors.

    U32 head;
    head  = clamp((int)(colors[refIdx.x].z * 31.0f + 0.5f), 0, 31);
    head |= clamp((int)(colors[refIdx.x].y * 63.0f + 0.5f - (F32)head * exp2(-5)), 0, 63) << 5;
    head |= clamp((int)(colors[refIdx.x].x * 31.0f + 0.5f - (F32)head * exp2(-11)), 0, 31) << 11;
    head |= clamp((int)(colors[refIdx.y].z * 31.0f + 0.5f - (F32)head * exp2(-16)), 0, 31) << 16;
    head |= clamp((int)(colors[refIdx.y].y * 63.0f + 0.5f - (F32)head * exp2(-21)), 0, 63) << 21;
    head |= clamp((int)(colors[refIdx.y].x * 31.0f + 0.5f - (F32)head * exp2(-27)), 0, 31) << 27;

    // Find the best lerp factor for each color.

    U32 bits = 0;
    Vec3f ref[4];
    decodeDXTColorHead(ref, head);

    for (int i = 0; i < num; i++)
    {
        int lerpIdx = 0;
        F32 lerpDist = (colors[i] - ref[0]).lenSqr();
        for (int j = 1; j < 4; j++)
        {
            F32 tmpDist = (colors[i] - ref[j]).lenSqr();
            if (tmpDist < lerpDist)
            {
                lerpIdx = j;
                lerpDist = tmpDist;
            }
        }
        bits |= lerpIdx << (indices[i] * 2);
    }
    return head | ((U64)bits << 32);
}

//------------------------------------------------------------------------

void FW::decodeDXTColors(Vec3f* colors, const U64& block)
{
    Vec3f ref[4];
    decodeDXTColorHead(ref, (U32)block);

    U32 bits = (U32)(block >> 32);
    for (int i = 0; i < 16; i++)
    {
        colors[i] = ref[bits & 3];
        bits >>= 2;
    }
}

//------------------------------------------------------------------------

U32 FW::encodeDXTNormalAxis(U32 headUV, const Vec3f& axis, int shift)
{
    int exponent = clamp((int)(floatToBits(axis.abs().max()) >> 23) + (-127 - 31 + 1 - 3 + 13), 0, 15);
    Vec3f scaled = axis * exp2(-exponent + (13 - 31));
    headUV |= exponent << shift;
    headUV |= (clamp((int)(scaled.z + 8.5f - (F32)headUV * exp2(-4 - shift)), 0, 15) ^ 8) << (4 + shift);
    headUV |= (clamp((int)(scaled.y + 8.5f - (F32)headUV * exp2(-8 - shift)), 0, 15) ^ 8) << (8 + shift);
    headUV |= (clamp((int)(scaled.x + 8.5f - (F32)headUV * exp2(-12 - shift)), 0, 15) ^ 8) << (12 + shift);
    return headUV;
}

//------------------------------------------------------------------------

Vec3f FW::decodeDXTNormalAxis(U32 headUV, int shift)
{
    S32 shifted = headUV << (16 - shift);
    int exponent = (shifted >> 16) & 15;
    return Vec3f((F32)shifted, (F32)(shifted << 4), (F32)(shifted << 8)) * exp2(exponent + (3 - 13));
}

//------------------------------------------------------------------------

static const F32 s_dxtNormalCoefs[4] = { -1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f };

void FW::encodeDXTNormals(U64& blockA, U64& blockB, const Vec3f* normals, const S32* indices, int num)
{
    // Use the average normal as base.

    FW_ASSERT(num > 0);
    Vec3f base = normals[0];
    for (int i = 1; i < num; i++)
        base += normals[i];

    // Degenerate => pick the input normal closest to the average.

    if (base.length() < 0.1f * (F32)num)
    {
        int baseIdx = 0;
        F32 baseDot = normals[0].dot(base);
        for (int i = 1; i < num; i++)
        {
            F32 tmpDot = normals[i].dot(base);
            if (tmpDot > baseDot)
            {
                baseIdx = i;
                baseDot = tmpDot;
            }
        }
        base = normals[baseIdx];
    }

    // Encode the base.

    U32 headBase = encodeRawNormal(base);
    base = Vec3f(decodeRawNormal(headBase));

    // Find the normal furthest away from the base.

    int uIdx = 0;
    F32 uDot = normals[0].dot(base);
    for (int i = 1; i < num; i++)
    {
        F32 tmpDot = normals[i].dot(base);
        if (tmpDot < uDot)
        {
            uIdx = i;
            uDot = tmpDot;
        }
    }

    // Encode the U axis.

    U32 headUV = encodeDXTNormalAxis(0, normals[uIdx] * base.length() - base, 0);
    Vec3f u = decodeDXTNormalAxis(headUV, 0);

    // Find the normal with the worst approximation so far.

    Vec3f uRef[4];
    for (int i = 0; i < 4; i++)
        uRef[i] = (base + u * s_dxtNormalCoefs[i]).normalized();

    Vec2i vIdx;
    F32 vDot = FW_F32_MAX;
    for (int i = 0; i < num; i++)
    {
        int lerpIdx = 0;
        F32 lerpDot = normals[i].dot(uRef[0]);
        for (int j = 1; j < 4; j++)
        {
            F32 tmpDot = normals[i].dot(uRef[j]);
            if (tmpDot > lerpDot)
            {
                lerpIdx = j;
                lerpDot = tmpDot;
            }
        }
        if (lerpDot < vDot)
        {
            vIdx = Vec2i(i, lerpIdx);
            vDot = lerpDot;
        }
    }

    // Encode as the V axis.

    Vec3f tmp = base + u * s_dxtNormalCoefs[vIdx.y];
    headUV = encodeDXTNormalAxis(headUV, normals[vIdx.x] * tmp.length() - tmp, 16);
    Vec3f v = decodeDXTNormalAxis(headUV, 16);

    // Find the best lerp factors for each input normal.

    Vec3f uvRef[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            uvRef[i + j * 4] = (base + u * s_dxtNormalCoefs[i] + v * s_dxtNormalCoefs[j]).normalized();

    U32 bitsU = 0;
    U32 bitsV = 0;
    for (int i = 0; i < num; i++)
    {
        int lerpIdx = 0;
        F32 lerpDot = normals[i].dot(uvRef[0]);
        for (int j = 1; j < 16; j++)
        {
            F32 tmpDot = normals[i].dot(uvRef[j]);
            if (tmpDot > lerpDot)
            {
                lerpIdx = j;
                lerpDot = tmpDot;
            }
        }
        int shift = indices[i] * 2;
        bitsU |= (lerpIdx & 3) << shift;
        bitsV |= (lerpIdx >> 2) << shift;
    }

    // Assemble the block.

    blockA = headBase | ((U64)bitsU << 32);
    blockB = headUV | ((U64)bitsV << 32);
}

//------------------------------------------------------------------------

void FW::decodeDXTNormals(Vec3f* normals, const U64& blockA, const U64& blockB)
{
    U32 headBase = (U32)blockA;
    U32 headUV   = (U32)blockB;
    U32 bitsU    = (U32)(blockA >> 32);
    U32 bitsV    = (U32)(blockB >> 32);

    Vec3f base = decodeRawNormal(headBase);
    Vec3f u = decodeDXTNormalAxis(headUV, 0);
    Vec3f v = decodeDXTNormalAxis(headUV, 16);

    for (int i = 0; i < 16; i++)
    {
        F32 cu = s_dxtNormalCoefs[(bitsU >> (i * 2)) & 3];
        F32 cv = s_dxtNormalCoefs[(bitsV >> (i * 2)) & 3];
        normals[i] = base + u * cu + v * cv;
    }
}

//------------------------------------------------------------------------

template <class S> __forceinline void FW::findMinMax(S x0, S x1, S x2, S& min, S& max)
{
    min = max = x0;
    if (x1 < min) min = x1;
    if (x1 > max) max = x1;
    if (x2 < min) min = x2;
    if (x2 > max) max = x2;
}

template <class S, class V> __forceinline bool FW::planeBoxOverlap(const V& normal, S d, const V& maxbox)
{
    int q;
    V vmin, vmax;
    for (q = 0; q <= 2; q++)
    {
        if(normal[q] > 0.0f)
        {
            vmin[q] = -maxbox[q];
            vmax[q] = maxbox[q];
        }
        else
        {
            vmin[q] = maxbox[q];
            vmax[q] = -maxbox[q];
        }
    }
    if (normal.dot(vmin) + d > 0.0f) return false;
    if (normal.dot(vmax) + d >= 0.0f) return true;
    return false;
}

#define AXISTEST_X01(a, b, fa, fb)                                          \
    p0 = a*v0.y - b*v0.z;                                                   \
    p2 = a*v2.y - b*v2.z;                                                   \
        if (p0 < p2) { min = p0; max = p2; } else { min = p2; max = p0; }   \
    rad = fa * boxHalfSize.y + fb * boxHalfSize.z;                          \
    if (min > rad || max < -rad) return false;

#define AXISTEST_X2(a, b, fa, fb)                                           \
    p0 = a*v0.y - b*v0.z;                                                   \
    p1 = a*v1.y - b*v1.z;                                                   \
        if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }   \
    rad = fa * boxHalfSize.y + fb * boxHalfSize.z;                          \
    if (min > rad || max < -rad) return false;

#define AXISTEST_Y02(a, b, fa, fb)                                          \
    p0 = -a*v0.x + b*v0.z;                                                  \
    p2 = -a*v2.x + b*v2.z;                                                  \
        if (p0 < p2) { min = p0; max = p2; } else { min = p2; max = p0; }   \
    rad = fa * boxHalfSize.x + fb * boxHalfSize.z;                          \
    if (min > rad || max < -rad) return false;

#define AXISTEST_Y1(a, b, fa, fb)                                           \
    p0 = -a*v0.x + b*v0.z;                                                  \
    p1 = -a*v1.x + b*v1.z;                                                  \
        if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }   \
    rad = fa * boxHalfSize.x + fb * boxHalfSize.z;                          \
    if (min > rad || max < -rad) return false;

#define AXISTEST_Z12(a, b, fa, fb)                                          \
    p1 = a*v1.x - b*v1.y;                                                   \
    p2 = a*v2.x - b*v2.y;                                                   \
        if (p2 < p1) { min = p2; max = p1; } else { min = p1; max = p2; }   \
    rad = fa * boxHalfSize.x + fb * boxHalfSize.y;                          \
    if (min > rad || max < -rad) return false;

#define AXISTEST_Z0(a, b, fa, fb)                                           \
    p0 = a*v0.x - b*v0.y;                                                   \
    p1 = a*v1.x - b*v1.y;                                                   \
        if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }   \
    rad = fa * boxHalfSize.x + fb * boxHalfSize.y;                          \
    if (min > rad || max < -rad) return false;

template <class S, class V> __forceinline bool FW::isectsDeltaTriangleBox(const V& p, const V& pu, const V& pv, const V& boxHalfSize)
{
  /*    use separating axis theorem to test overlap between triangle and box */
  /*    need to test for overlap in these directions: */
  /*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
  /*       we do not even need to test these) */
  /*    2) normal of the triangle */
  /*    3) crossproduct(edge from tri, {x,y,z}-direction) */
  /*       this gives 3x3=9 more tests */

   S min, max, p0, p1, p2, rad, fex, fey, fez;
   const V& v0 = p;
   V v1 = p + pu;
   V v2 = p + pv;

   /* Bullet 3:  */
   /*  test the 9 tests first (this was faster) */
   fex = abs(pu.x);
   fey = abs(pu.y);
   fez = abs(pu.z);
   AXISTEST_X01(pu.z, pu.y, fez, fey);
   AXISTEST_Y02(pu.z, pu.x, fez, fex);
   AXISTEST_Z12(pu.y, pu.x, fey, fex);

   V e = pv - pu;
   fex = abs(e.x);
   fey = abs(e.y);
   fez = abs(e.z);
   AXISTEST_X01(e.z, e.y, fez, fey);
   AXISTEST_Y02(e.z, e.x, fez, fex);
   AXISTEST_Z0(e.y, e.x, fey, fex);

   fex = abs(pv.x);
   fey = abs(pv.y);
   fez = abs(pv.z);
   AXISTEST_X2((-pv.z), (-pv.y), fez, fey);
   AXISTEST_Y1((-pv.z), (-pv.x), fez, fex);
   AXISTEST_Z12((-pv.y), (-pv.x), fey, fex);

   /* Bullet 1: */
   /*  first test overlap in the {x,y,z}-directions */
   /*  find min, max of the triangle each direction, and test for overlap in */
   /*  that direction -- this is equivalent to testing a minimal AABB around */
   /*  the triangle against the AABB */

   /* test in 0-direction */
   findMinMax(v0.x, v1.x, v2.x, min, max);
   if (min > boxHalfSize.x || max < -boxHalfSize.x) return false;

   /* test in 1-direction */
   findMinMax(v0.y, v1.y, v2.y, min, max);
   if (min > boxHalfSize.y || max < -boxHalfSize.y) return false;

   /* test in 2-direction */
   findMinMax(v0.z, v1.z, v2.z, min, max);
   if (min > boxHalfSize.z || max < -boxHalfSize.z) return false;

   /* Bullet 2: */
   /*  test if the box intersects the plane of the triangle */
   /*  compute plane equation of triangle: normal*x+d=0 */
   V normal = pv.cross(pu);
   S d = -normal.dot(v0); /* plane eq: normal.x+d=0 */
   return planeBoxOverlap(normal, d, boxHalfSize);
}

bool FW::isectsDeltaTriangleBox(const Vec3f& p, const Vec3f& pu, const Vec3f& pv, const Vec3f& boxHalfSize)
{
    return isectsDeltaTriangleBox<F32, Vec3f>(p, pu, pv, boxHalfSize);
}

bool FW::isectsDeltaTriangleBox(const Vec3d& p, const Vec3d& pu, const Vec3d& pv, const Vec3d& boxHalfSize)
{
    return isectsDeltaTriangleBox<F64, Vec3d>(p, pu, pv, boxHalfSize);
}

//------------------------------------------------------------------------

template <class S, class V2, class V3> __forceinline int FW::clipDeltaTriangleToBox(
    V2 baryOut[9], const V3& p, const V3& pu, const V3& pv, const V3& boxHalfSize)
{
    int num = 3;
    V2 baryTmp[7];
    baryTmp[0] = V2(0.0f, 0.0f);
    baryTmp[1] = V2(1.0f, 0.0f);
    baryTmp[2] = V2(0.0f, 1.0f);

    for (int axis = 0; axis < 3 && num; axis++)
    {
        const V2*       in      = (axis == 1) ? baryOut : baryTmp;
        V2*             out     = (axis == 1) ? baryTmp : baryOut;
        S               bias    = p[axis];
        V2              coef    = V2(pu[axis], pv[axis]);
        S               cut     = boxHalfSize[axis];
        int             numIn   = num;
        const V2*       bb      = &in[numIn - 1];
        S               bc      = bb->dot(coef) + bias;
        int             bs      = (bc < -cut) ? -1 : (bc > cut) ? 1 : 0;

        num = 0;
        for (int i = 0; i < numIn; i++)
        {
            const V2* ab = bb;
            S ac = bc;
            int as = bs;
            bb = &in[i];
            bc = bb->dot(coef) + bias;
            bs = (bc < -cut) ? -1 : (bc > cut) ? 1 : 0;

            if (as == 0)
                out[num++] = *ab;

            if (as != bs)
            {
                if (as != 0) out[num++] = lerp(*ab, *bb, clamp((ac - ((as < 0) ? -cut : cut)) / (ac - bc), (S)0.0f, (S)1.0f));
                if (bs != 0) out[num++] = lerp(*ab, *bb, clamp((ac - ((bs < 0) ? -cut : cut)) / (ac - bc), (S)0.0f, (S)1.0f));
            }
        }
    }
    return num;
}

int FW::clipDeltaTriangleToBox(Vec2f baryOut[9], const Vec3f& p, const Vec3f& pu, const Vec3f& pv, const Vec3f& boxHalfSize)
{
    return clipDeltaTriangleToBox<F32, Vec2f, Vec3f>(baryOut, p, pu, pv, boxHalfSize);
}

int FW::clipDeltaTriangleToBox(Vec2d baryOut[9], const Vec3d& p, const Vec3d& pu, const Vec3d& pv, const Vec3d& boxHalfSize)
{
    return clipDeltaTriangleToBox<F64, Vec2d, Vec3d>(baryOut, p, pu, pv, boxHalfSize);
}

//------------------------------------------------------------------------

F32 FW::quadrancePointToTri(const Vec3f& p, const Vec3f& a, const Vec3f& b)
{
    F32 aa = a.lenSqr();
    F32 bb = b.lenSqr();
    F32 pp = p.lenSqr();
    F32 ab = a.dot(b);
    F32 ap = a.dot(p);
    F32 bp = b.dot(p);

    // Interior.

    F32 d = aa * bb - ab * ab;
    F32 u = 0.0f;
    F32 v = 0.0f;

    if (d != 0.0f)
    {
        d = 1.0f / d;
        u = (bb * ap - ab * bp) * d;
        v = (aa * bp - ab * ap) * d;
    }

    if (u > 0.0f && v > 0.0f && u + v < 1.0f)
        return (a * u + b * v - p).lenSqr();

    // Edges.

    F32 res = FW_F32_MAX;
    if (u <= 0.0f)
    {
        F32 t = clamp(bp / bb, 0.0f, 1.0f);
        res = min(res, (bb * t - 2.0f * bp) * t + pp);
    }
    if (v <= 0.0f)
    {
        F32 t = clamp(ap / aa, 0.0f, 1.0f);
        res = min(res, (aa * t - 2.0f * ap) * t + pp);
    }
    if (u + v >= 1.0f)
    {
        Vec3f q = p - a;
        Vec3f c = b - a;
        F32 cc = c.lenSqr();
        F32 cq = c.dot(q);
        F32 t = clamp(cq / cc, 0.0f, 1.0f);
        res = min(res, (cc * t - 2.0f * cq) * t + q.lenSqr());
    }
    return res;
}

//------------------------------------------------------------------------

String FW::formatTime(F32 seconds)
{
    S64 stamp   = (S64)(seconds * 1000.0f + 0.5f);
    S32 millis  = (S32)(stamp % 1000);
    S32 secs    = (S32)((stamp / 1000) % 60);
    S32 mins    = (S32)((stamp / (60 * 1000)) % 60);
    S32 hours   = (S32)(stamp / (60 * 60 * 1000));

    if (hours)
        return sprintf("%02d:%02d:%02d.%03d", hours, mins, secs, millis);
    if (mins)
        return sprintf("%02d:%02d.%03d", mins, secs, millis);
    return sprintf("%d.%03ds", secs, millis);
}

//------------------------------------------------------------------------

S32 BitReader::read(int numBits)
{
    FW_ASSERT(m_ptr);
    FW_ASSERT(m_ofs >= 0 && m_ofs < 32);
    FW_ASSERT(numBits >= 0 && numBits <= 32);

    S32 value = *m_ptr >> m_ofs;
    m_ofs += numBits;

    if (m_ofs >= 32)
    {
        m_ptr++;
        m_ofs -= 32;
        if (m_ofs != 0)
        {
            int low = numBits - m_ofs;
            value &= (1 << low) - 1;
            value |= *m_ptr << low;
        }
    }

    if (numBits < 32)
        value &= (1 << numBits) - 1;
    return value;
}

//------------------------------------------------------------------------

void BitWriter::write(int numBits, S32 value)
{
    FW_ASSERT(m_array);
    FW_ASSERT(m_ofs >= 0 && m_ofs < 32);
    FW_ASSERT(numBits >= 0 && numBits <= 32);

    if (numBits < 32)
        value &= (1 << numBits) - 1;

    m_accum |= value << m_ofs;
    m_ofs += numBits;

    if (m_ofs >= 32)
    {
        m_array->add(m_accum);
        m_ofs -= 32;
        m_accum = (m_ofs == 0) ? 0 : (U32)value >> (numBits - m_ofs);
    }
}

//------------------------------------------------------------------------
