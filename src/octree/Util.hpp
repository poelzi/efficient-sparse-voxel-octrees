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

#pragma once
#include "base/Math.hpp"
#include "base/String.hpp"
#include "3d/ConvexPolyhedron.hpp"

namespace FW
{
//------------------------------------------------------------------------

Vec3i               getCubeChildPos         (const Vec3i& parentPos, int parentScale, int childIdx);
int                 getCubeChildIndex       (const Vec3i& childPos, int parentScale);
bool                isPosInCube             (const Vec3i& pos, const Vec3i& cubePos, int cubeScale);

U32                 encodeRawNormal         (const Vec3f& normal);
Vec3i               decodeRawNormal         (U32 value);

S32                 encodeContourNormal     (const Vec3f& normal);
S32                 encodeContourBounds     (S32 value, F32 lo, F32 hi);
Vec3f               decodeContourNormal     (S32 value);
Vec2f               decodeContourPosThick   (S32 value);
S32                 xformContourToChild     (S32 value, int childIdx);
bool                isectPolyWithContour    (ConvexPolyhedron& poly, S32 contour, int planeID = -1);

U64                 encodeDXTColors         (const Vec3f* colors, const S32* indices, int num);
void                decodeDXTColors         (Vec3f* colors, const U64& block);
void                encodeDXTNormals        (U64& blockA, U64& blockB, const Vec3f* normals, const S32* indices, int num); // the input must be normalized
void                decodeDXTNormals        (Vec3f* normals, const U64& blockA, const U64& blockB);

bool                isectsDeltaTriangleBox  (const Vec3f& p, const Vec3f& pu, const Vec3f& pv, const Vec3f& boxHalfSize);
bool                isectsDeltaTriangleBox  (const Vec3d& p, const Vec3d& pu, const Vec3d& pv, const Vec3d& boxHalfSize);
int                 clipDeltaTriangleToBox  (Vec2f baryOut[9], const Vec3f& p, const Vec3f& pu, const Vec3f& pv, const Vec3f& boxHalfSize);
int                 clipDeltaTriangleToBox  (Vec2d baryOut[9], const Vec3d& p, const Vec3d& pu, const Vec3d& pv, const Vec3d& boxHalfSize);
F32                 quadrancePointToTri     (const Vec3f& p, const Vec3f& a, const Vec3f& b);

String              formatTime              (F32 seconds);

inline const Vec3i& base2ToVec              (int idx);
inline const Vec3i& base3ToVec              (int idx);
inline const Vec3i& base4ToVec              (int idx);
inline int          base2ToBase3            (int idx);
inline int          base2ToBase4            (int idx);
inline int          base3ToBase4            (int idx);

//------------------------------------------------------------------------

class BitReader
{
public:
    explicit        BitReader   (const S32* ptr = NULL) : m_ptr(ptr), m_ofs(0) {}
                    BitReader   (const BitReader& other)    { operator=(other); }
                    ~BitReader  (void)                      {}

    const S32*      getPtr      (void) const                { return m_ptr; }
    int             getOfs      (void) const                { return m_ofs; }
    S32             read        (int numBits);

    BitReader&      operator=   (const BitReader& other)    { m_ptr = other.m_ptr; m_ofs = other.m_ofs; return *this; }

private:
    const S32*      m_ptr;
    S32             m_ofs;
};

//------------------------------------------------------------------------

class BitWriter
{
public:
    explicit        BitWriter   (Array<S32>* outArray = NULL) : m_array(outArray), m_accum(0), m_ofs(0) {}
                    BitWriter   (const BitWriter& other)    { operator=(other); }
                    ~BitWriter  (void)                      {}

    Array<S32>*     getArray    (void) const                { return m_array; }
    S32             getAccum    (void) const                { return m_accum; }
    int             getOfs      (void) const                { return m_ofs; }
    void            write       (int numBits, S32 value);

    BitWriter&      operator=   (const BitWriter& other)    { m_array = other.m_array; m_accum = other.m_accum; m_ofs = other.m_ofs; return *this; }

private:
    Array<S32>*     m_array;
    S32             m_accum;
    S32             m_ofs;
};

//------------------------------------------------------------------------

const Vec3i& base2ToVec(int idx)
{
    FW_ASSERT(idx >= 0 && idx < 8);
    static const Vec3i lut[8] =
    {
        Vec3i(0,0,0), Vec3i(1,0,0), Vec3i(0,1,0), Vec3i(1,1,0),
        Vec3i(0,0,1), Vec3i(1,0,1), Vec3i(0,1,1), Vec3i(1,1,1)
    };
    return lut[idx];
}

//------------------------------------------------------------------------

const Vec3i& base3ToVec(int idx)
{
    FW_ASSERT(idx >= 0 && idx < 27);
    static const Vec3i lut[27] =
    {
        Vec3i(0,0,0), Vec3i(1,0,0), Vec3i(2,0,0), Vec3i(0,1,0), Vec3i(1,1,0), Vec3i(2,1,0), Vec3i(0,2,0), Vec3i(1,2,0), Vec3i(2,2,0),
        Vec3i(0,0,1), Vec3i(1,0,1), Vec3i(2,0,1), Vec3i(0,1,1), Vec3i(1,1,1), Vec3i(2,1,1), Vec3i(0,2,1), Vec3i(1,2,1), Vec3i(2,2,1),
        Vec3i(0,0,2), Vec3i(1,0,2), Vec3i(2,0,2), Vec3i(0,1,2), Vec3i(1,1,2), Vec3i(2,1,2), Vec3i(0,2,2), Vec3i(1,2,2), Vec3i(2,2,2)
    };
    return lut[idx];
}

//------------------------------------------------------------------------

const Vec3i& base4ToVec(int idx)
{
    FW_ASSERT(idx >= 0 && idx < 64);
    static const Vec3i lut[64] =
    {
        Vec3i(0,0,0), Vec3i(1,0,0), Vec3i(2,0,0), Vec3i(3,0,0), Vec3i(0,1,0), Vec3i(1,1,0), Vec3i(2,1,0), Vec3i(3,1,0),
        Vec3i(0,2,0), Vec3i(1,2,0), Vec3i(2,2,0), Vec3i(3,2,0), Vec3i(0,3,0), Vec3i(1,3,0), Vec3i(2,3,0), Vec3i(3,3,0),
        Vec3i(0,0,1), Vec3i(1,0,1), Vec3i(2,0,1), Vec3i(3,0,1), Vec3i(0,1,1), Vec3i(1,1,1), Vec3i(2,1,1), Vec3i(3,1,1),
        Vec3i(0,2,1), Vec3i(1,2,1), Vec3i(2,2,1), Vec3i(3,2,1), Vec3i(0,3,1), Vec3i(1,3,1), Vec3i(2,3,1), Vec3i(3,3,1),
        Vec3i(0,0,2), Vec3i(1,0,2), Vec3i(2,0,2), Vec3i(3,0,2), Vec3i(0,1,2), Vec3i(1,1,2), Vec3i(2,1,2), Vec3i(3,1,2),
        Vec3i(0,2,2), Vec3i(1,2,2), Vec3i(2,2,2), Vec3i(3,2,2), Vec3i(0,3,2), Vec3i(1,3,2), Vec3i(2,3,2), Vec3i(3,3,2),
        Vec3i(0,0,3), Vec3i(1,0,3), Vec3i(2,0,3), Vec3i(3,0,3), Vec3i(0,1,3), Vec3i(1,1,3), Vec3i(2,1,3), Vec3i(3,1,3),
        Vec3i(0,2,3), Vec3i(1,2,3), Vec3i(2,2,3), Vec3i(3,2,3), Vec3i(0,3,3), Vec3i(1,3,3), Vec3i(2,3,3), Vec3i(3,3,3)
    };
    return lut[idx];
}

//------------------------------------------------------------------------

int base2ToBase3(int idx)
{
    FW_ASSERT(idx >= 0 && idx < 8);
    static const S32 lut[8] = { 0, 1, 3, 4, 9, 10, 12, 13 };
    return lut[idx];
}

//------------------------------------------------------------------------

int base2ToBase4(int idx)
{
    FW_ASSERT(idx >= 0 && idx < 8);
    static const S32 lut[8] = { 0, 1, 4, 5, 16, 17, 20, 21 };
    return lut[idx];
}

//------------------------------------------------------------------------

int base3ToBase4(int idx)
{
    FW_ASSERT(idx >= 0 && idx < 27);
    static const S32 lut[27] =
    {
        0, 1, 2, 4, 5, 6, 8, 9, 10,
        16, 17, 18, 20, 21, 22, 24, 25, 26,
        32, 33, 34, 36, 37, 38, 40, 41, 42
    };
    return lut[idx];
}

//------------------------------------------------------------------------
}
