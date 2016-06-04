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
#include "3d/Texture.hpp"
#include "../Util.hpp"

namespace FW
{
//------------------------------------------------------------------------

class DisplacementMap
{
public:
    struct Texel
    {
        F32             height;
        Vec2f           grad;
    };

    struct Bounds
    {
        Vec2f           heightGrad;
        F32             heightAvg;
        F32             heightDiff;
        Vec2f           gradAvg;
        Vec2f           gradDiff;
    };

public:
                        DisplacementMap     (const Texture& tex, F32 coef, F32 bias);
                        ~DisplacementMap    (void);

    const Vec2i&        getSize             (void) const        { return m_size; }
    int                 getNumLevels        (void) const        { return m_bounds.getSize(); }

    const Texel&        getTexel            (int x, int y) const;
    const Bounds&       getBounds           (int x, int y, int level) const;

private:
                        DisplacementMap     (DisplacementMap&); // forbidden
    DisplacementMap&    operator=           (DisplacementMap&); // forbidden

private:
    Vec2i               m_size;
    U8*                 m_data;
    Texel*              m_texels;
    Array<Bounds*>      m_bounds;
};

//------------------------------------------------------------------------

class DisplacedTriangle
{
public:
    struct Temp
    {
        Array<Vec3i>    stack;
        Array<Vec3i>    rects; // Vec3i(x, y, scale) - negative scale indicates clipped texel
        Array<Vec2f>    verts;
        F32             area;
    };

    struct Normal
    {
        Vec3f           avg;
        Vec3f           lo;
        Vec3f           hi;
    };

private:
    struct DecodedIntersection
    {
        S32             numRects;
        S32             numVerts;
        const Vec3i*    rects;
        const Vec2f*    verts;
        const S32*      next;
    };

    struct Quadrangle
    {
        Vec3f           p0;
        Vec3f           d1;
        Vec3f           d2;
        Vec3f           d3;
    };

public:
                        DisplacedTriangle   (void);
                        ~DisplacedTriangle  (void);

    void                set                 (DisplacementMap* map,
                                             const Vec3f& p, const Vec3f& pu, const Vec3f& pv,
                                             const Vec3f& n, const Vec3f& nu, const Vec3f& nv,
                                             const Vec2f& t, const Vec2f& tu, const Vec2f& tv);

    void                getInitialIsect     (Array<S32>& isectOut, Temp& temp) const;
    F32                 intersectBox        (Array<S32>& isectOut, const S32* isectIn, const Vec3f& boxMid, const Vec3f& boxHalfSize, Temp& temp) const; // returns area

    static const S32*   getNextIsect        (const S32* isectIn)    { DecodedIntersection isect; decodeIntersection(isect, isectIn); return isect.next; }
    void                exportIsect         (BitWriter& out, const S32* isectIn) const;
    void                importIsect         (Array<S32>& isectOut, BitReader& in, Temp& temp) const;

    void                getTexAndNormal     (Vec2f& texLo, Vec2f& texHi, Normal& normal, const S32* isectIn) const;
    void                getPlaneNormals     (Array<Vec3f>& res, const S32* isectIn) const;
    void                expandPlaneBounds   (Vec2f& bounds, const S32* isectIn, const Vec3f& normal, const Vec3f& origin) const;
    void                removeNearbyProbes  (Array<Vec3f>& probes, const S32* isectIn, F32 distSqr) const;

//  DisplacedTriangle&  operator=           (DisplacedTriangle&); // default

private:
    static Vec3f        rectToT             (const Vec3i& r)        { Vec3f t((F32)r.x, (F32)r.y, 0.0f); t += (F32)(1 << max(r.z, 0)) * 0.5f; return t; }
    Vec3f               mapBary             (F32 s, F32 t) const    { Vec2f uv = m_b + m_bs * s + m_bt * t; return Vec3f(uv.x, uv.y, 1.0f - uv.x - uv.y); }

    static void         clearTemp           (Temp& temp)            { temp.stack.clear(); temp.rects.clear(); temp.verts.clear(); temp.area = 0.0f; }
    static void         encodeIntersection  (Array<S32>& isectOut, const Temp& temp);
    static void         decodeIntersection  (DecodedIntersection& out, const S32* ptr);

    bool                intersectBoxBounds  (Temp& out, const Vec3f& boxMid, const Vec3f& boxHalfSize, const Vec3i& rect) const; // false = subdivide
    void                intersectBoxTexel   (Temp& out, const Vec3f& boxMid, const Vec3f& boxHalfSize, int x, int y) const;
    void                addVerts            (Temp& out, const Vec2f& t, const Vec2f& tu, const Vec2f& tv, int numBary, const Vec2f* bary) const;
    static F32          decFloat            (F32 v);

    void                getNormalizedBounds (DisplacementMap::Bounds& res, const Vec3i& rect) const;
    void                getTexelQuadrangle  (Quadrangle& res, int x, int y) const;
    void                updateNormal        (Normal& res, const Vec3f& t, const Vec2f& g, const Vec2f& ga, const Vec2f& gd, F32 weight) const;

private:
                        DisplacedTriangle   (DisplacedTriangle&); // forbidden

private:
    DisplacementMap*    m_map;
    Vec2f               m_b;
    Vec2f               m_bs;
    Vec2f               m_bt;
    Vec3f               m_p;
    Vec3f               m_ps;
    Vec3f               m_pt;
    Vec3f               m_n;
    Vec3f               m_ns;
    Vec3f               m_nt;
    bool                m_constantNormal;
    F32                 m_minNormalLen;
    Vec2i               m_boundLo;
    Vec2i               m_boundHi;
    S32                 m_posBits;
    S32                 m_levelBits;
};

//------------------------------------------------------------------------
}
