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

#include "DisplacementMap.hpp"
#include "../Util.hpp"

using namespace FW;

//------------------------------------------------------------------------

#define MAX_APPROXIMATION_ERROR 1.0f // voxels

//------------------------------------------------------------------------

static inline Vec2f vecMin(const Vec2f& a, const Vec2f& b) { return Vec2f(fastMin(a.x, b.x), fastMin(a.y, b.y)); }
static inline Vec2f vecMax(const Vec2f& a, const Vec2f& b) { return Vec2f(fastMax(a.x, b.x), fastMax(a.y, b.y)); }
static inline Vec3f vecMin(const Vec3f& a, const Vec3f& b) { return Vec3f(fastMin(a.x, b.x), fastMin(a.y, b.y), fastMin(a.z, b.z)); }
static inline Vec3f vecMax(const Vec3f& a, const Vec3f& b) { return Vec3f(fastMax(a.x, b.x), fastMax(a.y, b.y), fastMax(a.z, b.z)); }

//------------------------------------------------------------------------

DisplacementMap::DisplacementMap(const Texture& tex, F32 coef, F32 bias)
{
    const Image* image = tex.getImage();
    FW_ASSERT(image);
    m_size = image->getSize();
    FW_ASSERT(m_size.min() > 0);

    // Count levels.

    int numLevels = 1;
    while ((1 << (numLevels - 1)) < m_size.max())
        numLevels++;

    // Allocate data.

    int numTexels = m_size.x * m_size.y;
    m_data = new U8[numTexels * (sizeof(Texel) + numLevels * sizeof(Bounds))];
    m_texels = (Texel*)m_data;
    for (int i = 0; i < numLevels; i++)
        m_bounds.add((Bounds*)(m_texels + numTexels) + numTexels * i);

    // Convert heights.

    Image converted(m_size, ImageFormat::ABGR_8888);
    converted = *image;
    const U32* src = (const U32*)converted.getPtr();
    for (int i = 0; i < numTexels; i++)
        m_texels[i].height = (F32)((src[i] >> 8) & 0xFF) * coef + bias;

    // Compute gradients.

    for (int y = 0; y < m_size.y; y++)
    {
        Texel* dst = m_texels + m_size.x * y;
        for (int x = 0; x < m_size.x; x++)
        {
            dst->grad.x = (getTexel(x + 1, y).height - getTexel(x - 1, y).height) * (2.0f / 8.0f) +
                          (getTexel(x + 1, y + 1).height - getTexel(x - 1, y + 1).height) * (1.0f / 8.0f) +
                          (getTexel(x + 1, y - 1).height - getTexel(x - 1, y - 1).height) * (1.0f / 8.0f);
            dst->grad.y = (getTexel(x, y + 1).height - getTexel(x, y - 1).height) * (2.0f / 8.0f) +
                          (getTexel(x + 1, y + 1).height - getTexel(x + 1, y - 1).height) * (1.0f / 8.0f) +
                          (getTexel(x - 1, y + 1).height - getTexel(x - 1, y - 1).height) * (1.0f / 8.0f);
            dst++;
        }
    }

    // Generate bounds for the first level.

    for (int y = 0; y < m_size.y; y++)
    {
        Bounds* dst = m_bounds[0] + m_size.x * y;
        for (int x = 0; x < m_size.x; x++)
        {
            const Texel& t00 = getTexel(x, y);
            const Texel& t10 = getTexel(x + 1, y);
            const Texel& t01 = getTexel(x, y + 1);
            const Texel& t11 = getTexel(x + 1, y + 1);

            Vec2f g = Vec2f(t10.height - t00.height + t11.height - t01.height, t01.height - t00.height + t11.height - t10.height) * 0.25f;
            Vec2f glo = vecMin(vecMin(vecMin(t00.grad, t10.grad), t01.grad), t11.grad);
            Vec2f ghi = vecMax(vecMax(vecMax(t00.grad, t10.grad), t01.grad), t11.grad);
            F32 blo = min(t00.height + g.x + g.y, t10.height - g.x + g.y, t01.height + g.x - g.y, t11.height - g.x - g.y);
            F32 bhi = max(t00.height + g.x + g.y, t10.height - g.x + g.y, t01.height + g.x - g.y, t11.height - g.x - g.y);

            dst->heightGrad = g * 2.0f;
            dst->heightAvg = (blo + bhi) * 0.5f;
            dst->heightDiff = (bhi - blo) * 0.5f;
            dst->gradAvg = (glo + ghi) * 0.5f;
            dst->gradDiff = (ghi - glo) * 0.5f;
            dst++;
        }
    }

    // Generate bounds for the remaining levels.

    for (int i = 1; i < numLevels; i++)
    {
        int step = 1 << (i - 1);
        F32 coef = (F32)step * 0.5f;

        for (int y = 0; y < m_size.y; y++)
        {
            Bounds* dst = m_bounds[i] + m_size.x * y;
            for (int x = 0; x < m_size.x; x++)
            {
                const Bounds& b00 = getBounds(x, y, i - 1);
                const Bounds& b10 = getBounds(x + step, y, i - 1);
                const Bounds& b01 = getBounds(x, y + step, i - 1);
                const Bounds& b11 = getBounds(x + step, y + step, i - 1);

                Vec2f g = (b00.heightGrad + b10.heightGrad + b01.heightGrad + b11.heightGrad) * 0.25f;
                Vec2f glo = vecMin(vecMin(vecMin(b00.gradAvg - b00.gradDiff, b10.gradAvg - b10.gradDiff), b01.gradAvg - b01.gradDiff), b11.gradAvg - b11.gradDiff);
                Vec2f ghi = vecMax(vecMax(vecMax(b00.gradAvg + b00.gradDiff, b10.gradAvg + b10.gradDiff), b01.gradAvg + b01.gradDiff), b11.gradAvg + b11.gradDiff);
                F32 a00 = b00.heightAvg + (+g.x+g.y) * coef;
                F32 a10 = b10.heightAvg + (-g.x+g.y) * coef;
                F32 a01 = b01.heightAvg + (+g.x-g.y) * coef;
                F32 a11 = b11.heightAvg + (-g.x-g.y) * coef;
                F32 d00 = b00.heightDiff + (b00.heightGrad - g).abs().dot(Vec2f(coef));
                F32 d10 = b10.heightDiff + (b10.heightGrad - g).abs().dot(Vec2f(coef));
                F32 d01 = b01.heightDiff + (b01.heightGrad - g).abs().dot(Vec2f(coef));
                F32 d11 = b11.heightDiff + (b11.heightGrad - g).abs().dot(Vec2f(coef));
                F32 blo = min(a00 - d00, a10 - d10, a01 - d01, a11 - d11);
                F32 bhi = max(a00 + d00, a10 + d10, a01 + d01, a11 + d11);

                dst->heightGrad = g;
                dst->heightAvg = (blo + bhi) * 0.5f;
                dst->heightDiff = (bhi - blo) * 0.5f;
                dst->gradAvg = (glo + ghi) * 0.5f;
                dst->gradDiff = (ghi - glo) * 0.5f;
                dst++;
            }
        }
    }
}

//------------------------------------------------------------------------

DisplacementMap::~DisplacementMap(void)
{
    delete[] m_data;
}

//------------------------------------------------------------------------

const DisplacementMap::Texel& DisplacementMap::getTexel(int x, int y) const
{
    x %= m_size.x;
    y %= m_size.y;
    if (x < 0) x += m_size.x;
    if (y < 0) y += m_size.y;
    return m_texels[x + y * m_size.x];
}

//------------------------------------------------------------------------

const DisplacementMap::Bounds& DisplacementMap::getBounds(int x, int y, int level) const
{
    x %= m_size.x;
    y %= m_size.y;
    if (x < 0) x += m_size.x;
    if (y < 0) y += m_size.y;
    level = clamp(level, 0, m_bounds.getSize() - 1);
    return m_bounds[level][x + y * m_size.x];
}

//------------------------------------------------------------------------

DisplacedTriangle::DisplacedTriangle(void)
:   m_map   (NULL)
{
}

//------------------------------------------------------------------------

DisplacedTriangle::~DisplacedTriangle(void)
{
}

//------------------------------------------------------------------------

void DisplacedTriangle::set(
    DisplacementMap* map,
    const Vec3f& p, const Vec3f& pu, const Vec3f& pv,
    const Vec3f& n, const Vec3f& nu, const Vec3f& nv,
    const Vec2f& t, const Vec2f& tu, const Vec2f& tv)
{
    FW_ASSERT(map);
    m_map = map;

    // Create mapping from texels to barycentric coordinates.

    Mat3f bToT;
    bToT.setCol(0, Vec3f(tu, 0.0f));
    bToT.setCol(1, Vec3f(tv, 0.0f));
    bToT.setCol(2, Vec3f(t, 1.0f));
    bToT = Mat3f::scale(Vec2f((F32)map->getSize().x, (F32)map->getSize().y)) * bToT;
    bToT = Mat3f::translate(Vec2f(-0.5f)) * bToT;
    Mat3f tToB = bToT.inverted();
    m_b = Vec3f(tToB.getCol(2)).getXY();
    m_bs = Vec3f(tToB.getCol(0)).getXY();
    m_bt = Vec3f(tToB.getCol(1)).getXY();

    // Create mapping from texels to positions on the triangle.

    Mat3f bToP;
    bToP.setCol(0, pu);
    bToP.setCol(1, pv);
    bToP.setCol(2, p);
    Mat3f tToP = bToP * tToB;
    m_p = tToP.getCol(2);
    m_ps = tToP.getCol(0);
    m_pt = tToP.getCol(1);

    // Create mapping from texels to normal vectors.

    Mat3f bToN;
    bToN.setCol(0, nu);
    bToN.setCol(1, nv);
    bToN.setCol(2, n);
    Mat3f tToN = bToN * tToB;
    m_n = tToN.getCol(2);
    m_ns = tToN.getCol(0);
    m_nt = tToN.getCol(1);
    m_minNormalLen = sqrt(max(quadrancePointToTri(-n, nu, nv), 1.0e-8f));

    // Detect constant normal.

    Vec3f n0 = n.normalized();
    Vec3f n1 = (n + nu).normalized();
    Vec3f n2 = (n + nv).normalized();
    m_constantNormal = (min(n0.dot(n1), n0.dot(n2)) >= 0.999f);

    if (m_constantNormal)
    {
        m_n = n0;
        m_ns = 0.0f;
        m_nt = 0.0f;
        m_minNormalLen = 1.0f;
    }

    // Determine bounding rectangle.

    Vec3f t0 = bToT.getCol(2);
    Vec3f t1 = t0 + bToT.getCol(0);
    Vec3f t2 = t0 + bToT.getCol(1);
    m_boundLo = Vec2i((int)floor(min(t0.x, t1.x, t2.x)), (int)floor(min(t0.y, t1.y, t2.y)));
    m_boundHi = Vec2i((int)ceil(max(t0.x, t1.x, t2.x)), (int)ceil(max(t0.y, t1.y, t2.y)));

    // Count the number of bits needed to encode rectangle position.

    m_posBits = 0;
    while (m_posBits < 30 && (1 << m_posBits) < (m_boundHi - m_boundLo).max())
        m_posBits++;

    // Count the number of bits needed to encode rectangle scale.

    m_levelBits = 1;
    while ((1 << m_levelBits) <= m_posBits)
        m_levelBits++;
}

//------------------------------------------------------------------------

void DisplacedTriangle::getInitialIsect(Array<S32>& isectOut, Temp& temp) const
{
    clearTemp(temp);
    temp.rects.add(Vec3i(m_boundLo, m_posBits));
    encodeIntersection(isectOut, temp);
}

//------------------------------------------------------------------------

F32 DisplacedTriangle::intersectBox(Array<S32>& isectOut, const S32* isectIn, const Vec3f& boxMid, const Vec3f& boxHalfSize, Temp& temp) const
{
    DecodedIntersection isect;
    decodeIntersection(isect, isectIn);

    clearTemp(temp);
    temp.stack.set(isect.rects, isect.numRects);

    while (temp.stack.getSize())
    {
        Vec3i rect = temp.stack.removeLast();
        if (rect.z <= 0)
        {
            intersectBoxTexel(temp, boxMid, boxHalfSize, rect.x, rect.y);
            continue;
        }
        if (intersectBoxBounds(temp, boxMid, boxHalfSize, rect))
            continue;

        int half = 1 << (rect.z - 1);
        for (int i = 3; i >= 0; i--)
        {
            Vec3i child(rect.x + half * (i & 1), rect.y + half * (i >> 1), rect.z - 1);
            if (child.x < m_boundHi.x && child.y < m_boundHi.y)
                temp.stack.add(child);
        }
    }

    encodeIntersection(isectOut, temp);
    return temp.area;
}

//------------------------------------------------------------------------

void DisplacedTriangle::exportIsect(BitWriter& out, const S32* isectIn) const
{
    DecodedIntersection isect;
    decodeIntersection(isect, isectIn);

    for (int i = 0; i < isect.numRects; i++)
    {
        out.write(1, 1);
        out.write(m_posBits, isect.rects[i].x - m_boundLo.x);
        out.write(m_posBits, isect.rects[i].y - m_boundLo.y);
        out.write(m_levelBits, max(isect.rects[i].z, 0));
    }

    out.write(1, 0);
}

//------------------------------------------------------------------------

void DisplacedTriangle::importIsect(Array<S32>& isectOut, BitReader& in, Temp& temp) const
{
    clearTemp(temp);
    while (in.read(1))
    {
        Vec3i& r = temp.rects.add();
        r.x = in.read(m_posBits) + m_boundLo.x;
        r.y = in.read(m_posBits) + m_boundLo.y;
        r.z = in.read(m_levelBits);
    }
    encodeIntersection(isectOut, temp);
}

//------------------------------------------------------------------------

void DisplacedTriangle::getTexAndNormal(Vec2f& texLo, Vec2f& texHi, Normal& normal, const S32* isectIn) const
{
    DecodedIntersection isect;
    decodeIntersection(isect, isectIn);

    int numClip = 0;
    texLo = +FW_F32_MAX;
    texHi = -FW_F32_MAX;
    normal.avg = 0.0f;
    normal.lo = 1.0f;
    normal.hi = -1.0f;

    // Rectangles.

    for (int i = 0; i < isect.numRects; i++)
    {
        const Vec3i& rect = isect.rects[i];
        if (rect.z < 0)
        {
            numClip++;
            continue;
        }

        Vec3f t = rectToT(rect);
        texLo.x = min(texLo.x, t.x - t.z);
        texLo.y = min(texLo.y, t.y - t.z);
        texHi.x = max(texHi.x, t.x + t.z);
        texHi.y = max(texHi.y, t.y + t.z);

        const DisplacementMap::Bounds& b = m_map->getBounds(rect.x, rect.y, rect.z);
        updateNormal(normal, t, b.heightGrad, b.gradAvg, b.gradDiff, sqr(t.z));
    }

    // Vertices.

    F32 normalWeight = (F32)numClip / (F32)isect.numVerts;
    for (int i = 0; i < isect.numVerts; i++)
    {
        const Vec2f& t = isect.verts[i];
        Vec2f tfloor(floor(t.x), floor(t.y));
        Vec2i xy((S32)tfloor.x, (S32)tfloor.y);
        Vec2f f = t - tfloor;

        texLo.x = min(texLo.x, t.x);
        texLo.y = min(texLo.y, t.y);
        texHi.x = max(texHi.x, t.x);
        texHi.y = max(texHi.y, t.y);

        const DisplacementMap::Texel& t00 = m_map->getTexel(xy.x, xy.y);
        const DisplacementMap::Texel& t10 = m_map->getTexel(xy.x + 1, xy.y);
        const DisplacementMap::Texel& t01 = m_map->getTexel(xy.x, xy.y + 1);
        const DisplacementMap::Texel& t11 = m_map->getTexel(xy.x + 1, xy.y + 1);

        Vec2f g = lerp(lerp(t00.grad, t10.grad, f.x), lerp(t01.grad, t11.grad, f.x), f.y);
        updateNormal(normal, Vec3f(t, 0.0f), g, g, Vec2f(0.0f), normalWeight);
    }

    FW_ASSERT(m_map);
    Vec2i size = m_map->getSize();
    Vec2f coef(1.0f / (F32)size.x, 1.0f / (F32)size.y);
    texLo = (texLo + 0.5f) * coef;
    texHi = (texHi + 0.5f) * coef;
    normal.avg = normal.avg.normalized();
}

//------------------------------------------------------------------------

void DisplacedTriangle::getPlaneNormals(Array<Vec3f>& res, const S32* isectIn) const
{
    FW_ASSERT(m_map);
    DecodedIntersection isect;
    decodeIntersection(isect, isectIn);

    // Rectangles.

    for (int i = 0; i < isect.numRects; i++)
    {
        const Vec3i& rect = isect.rects[i];
        if (rect.z < 0)
            continue;

        const DisplacementMap::Bounds& b = m_map->getBounds(rect.x, rect.y, rect.z);
        Vec3f t = rectToT(rect);

        if (m_constantNormal)
            res.add((m_ps + m_n * b.heightGrad.x).cross(m_pt + m_n * b.heightGrad.y));
        else
        {
            Vec3f n = m_n + m_ns * t.x + m_nt * t.y;
            F32 c = 1.0f / n.length();
            Vec3f rs = m_ps + (n * b.heightGrad.x + m_ns * b.heightAvg) * c;
            Vec3f rt = m_pt + (n * b.heightGrad.y + m_nt * b.heightAvg) * c;
            res.add(rs.cross(rt));
        }
    }

    // Vertices.

    int prevId = -1;
    for (int i = 0; i < isect.numVerts; i++)
    {
        const Vec2f& t = isect.verts[i];
        Vec2f tfloor(floor(t.x), floor(t.y));
        Vec2i xy((S32)floor(t.x), (S32)floor(t.y));
        bool isFirstTri = (t.x - tfloor.x >= t.y - tfloor.y);

        int id = (xy.x + xy.y * m_map->getSize().x) * 2 + ((isFirstTri) ? 0 : 1);
        if (id == prevId)
            continue;
        prevId = id;

        Quadrangle q;
        getTexelQuadrangle(q, xy.x, xy.y);
        if (isFirstTri)
            res.add(q.d1.cross(q.d3));
        else
            res.add(q.d3.cross(q.d2));
    }
}

//------------------------------------------------------------------------

void DisplacedTriangle::expandPlaneBounds(Vec2f& bounds, const S32* isectIn, const Vec3f& normal, const Vec3f& origin) const
{
    DecodedIntersection isect;
    decodeIntersection(isect, isectIn);

    // Rectangles.

    for (int i = 0; i < isect.numRects; i++)
    {
        const Vec3i& rect = isect.rects[i];
        if (rect.z < 0)
            continue;

        DisplacementMap::Bounds b;
        getNormalizedBounds(b, rect);

        Vec3f t = rectToT(rect);
        F32 psd = normal.dot(m_ps);
        F32 ptd = normal.dot(m_pt);
        F32 nd = normal.dot(m_n);
        F32 avg = normal.dot(m_p - origin) + psd * t.x + ptd * t.y;
        F32 diff;

        if (m_constantNormal)
            diff = abs(nd) * b.heightDiff + (abs(nd * b.heightGrad.x + psd) + abs(nd * b.heightGrad.y + ptd)) * t.z;
        else
        {
            F32 nsd = normal.dot(m_ns);
            F32 ntd = normal.dot(m_nt);
            Vec2f bhi = b.heightGrad.abs() * t.z + b.heightDiff;

            F32 tmp = abs(nd * b.heightGrad.x + nsd * b.heightAvg + psd);
            tmp += abs(nd * b.heightGrad.y + ntd * b.heightAvg + ptd);
            tmp += abs(nsd) * bhi.x + abs(ntd) * bhi.y + abs(nsd * b.heightGrad.y + ntd * b.heightGrad.x) * t.z;

            nd += nsd * t.x + ntd * t.y;
            diff = abs(nd) * b.heightDiff + tmp * t.z;
        }

        avg += nd * b.heightAvg;
        bounds.x = fastMin(bounds.x, avg - diff);
        bounds.y = fastMax(bounds.y, avg + diff);
    }

    // Vertices.

    for (int i = 0; i < isect.numVerts; i++)
    {
        const Vec2f& t = isect.verts[i];
        Vec2f tfloor(floor(t.x), floor(t.y));
        Vec2f f = t - tfloor;

        Quadrangle q;
        getTexelQuadrangle(q, (S32)tfloor.x, (S32)tfloor.y);

        F32 d = normal.dot(q.p0 - origin);
        if (f.x >= f.y)
            d += normal.dot(q.d1) * (f.x - f.y) + normal.dot(q.d3) * f.y;
        else
            d += normal.dot(q.d3) * f.x + normal.dot(q.d2) * (f.y - f.x);

        bounds.x = fastMin(bounds.x, d);
        bounds.y = fastMax(bounds.y, d);
    }
}

//------------------------------------------------------------------------

void DisplacedTriangle::removeNearbyProbes(Array<Vec3f>& probes, const S32* isectIn, F32 distSqr) const
{
    DecodedIntersection isect;
    decodeIntersection(isect, isectIn);

    for (int i = 0; i < isect.numRects; i++)
    {
        const Vec3i& rect = isect.rects[i];
        Quadrangle q;
        F32 limit = distSqr;

        // Texel => exact quadrangle.

        if (rect.z < 0)
            getTexelQuadrangle(q, rect.x, rect.y);

        // Bounds => approximate quadrangle.

        else
        {
            DisplacementMap::Bounds b;
            getNormalizedBounds(b, rect);

            Vec3f t = rectToT(rect);
            Vec3f n = m_n + m_ns * t.x + m_nt * t.y;
            Vec3f rs = (m_ps + n * b.heightGrad.x) * t.z;
            Vec3f rt = (m_pt + n * b.heightGrad.y) * t.z;
            Vec3f err = n.abs() * b.heightDiff;

            if (!m_constantNormal)
            {
                Vec3f ns = m_ns * t.z;
                Vec3f nt = m_nt * t.z;
                rs += ns * b.heightAvg;
                rt += nt * b.heightAvg;
                err += ns.abs() * (b.heightDiff + abs(b.heightGrad.x) * t.z);
                err += nt.abs() * (b.heightDiff + abs(b.heightGrad.y) * t.z);
                err += (ns * b.heightGrad.y + nt * b.heightGrad.x).abs() * t.z;
            }

            q.p0 = m_p + m_ps * t.x + m_pt * t.y + n * b.heightAvg - rs - rt;
            q.d1 = rs * 2.0f;
            q.d2 = rt * 2.0f;
            q.d3 = q.d1 + q.d2;

            limit = sqrt(limit) - err.length();
            if (limit <= 0.0f)
                continue;
            limit = sqr(limit);
        }

        // Check points against the quadrangle.

        for (int j = probes.getSize() - 1; j >= 0; j--)
        {
            Vec3f t = probes[j] - q.p0;
            if (quadrancePointToTri(t, q.d1, q.d3) <= limit || quadrancePointToTri(t, q.d3, q.d2) <= limit)
                probes.removeSwap(j);
        }
    }
}

//------------------------------------------------------------------------

void DisplacedTriangle::encodeIntersection(Array<S32>& isectOut, const Temp& temp)
{
    if (!temp.rects.getSize())
    {
        FW_ASSERT(!temp.verts.getSize());
        return;
    }

    isectOut.add(temp.rects.getSize());
    isectOut.add(temp.verts.getSize());
    isectOut.add((S32*)temp.rects.getPtr(), temp.rects.getSize() * 3);
    isectOut.add((S32*)temp.verts.getPtr(), temp.verts.getSize() * 2);
}

//------------------------------------------------------------------------

void DisplacedTriangle::decodeIntersection(DecodedIntersection& out, const S32* ptr)
{
    out.numRects = *ptr++;
    out.numVerts = *ptr++;
    out.rects = (const Vec3i*)ptr;
    ptr += out.numRects * 3;
    out.verts = (const Vec2f*)ptr;
    ptr += out.numVerts * 2;
    out.next = ptr;
}

//------------------------------------------------------------------------

bool DisplacedTriangle::intersectBoxBounds(Temp& out, const Vec3f& boxMid, const Vec3f& boxHalfSize, const Vec3i& rect) const
{
    // Barys are outside the triangle => miss.

    Vec3f t = rectToT(rect);
    Vec3f bd = Vec3f(m_bs.abs() + m_bt.abs(), abs(m_bs.x + m_bs.y) + abs(m_bt.x + m_bt.y)) * t.z;
    if ((mapBary(t.x, t.y) + bd).min() < 0.0f)
        return true;

    // Bounds.

    DisplacementMap::Bounds b;
    getNormalizedBounds(b, rect);

    // Quadratic function of displaced position.

    Vec3f n  = m_n + m_ns * t.x + m_nt * t.y;
    Vec3f r  = n * b.heightAvg + m_p + m_ps * t.x + m_pt * t.y - boxMid;
    Vec3f rs = (n * b.heightGrad.x + m_ps) * t.z;
    Vec3f rt = (n * b.heightGrad.y + m_pt) * t.z;

    Vec3f quad = 0.0f;
    if (m_constantNormal)
    {
        // AABB is outside the box => miss.

        Vec3f aabb = rs.abs() + rt.abs() + n.abs() * b.heightDiff;
        if ((r.abs() - aabb - boxHalfSize).max() > 0.0f)
            return true;

        // Box is outside the OBB => miss.

        Vec3f fs = n.cross(rt);
        F32 tdet = abs(fs.dot(rs));
        if (abs(fs.dot(r)) - tdet > fs.abs().dot(boxHalfSize))
            return true;
        Vec3f ft = n.cross(rs);
        if (abs(ft.dot(r)) - tdet > ft.abs().dot(boxHalfSize))
            return true;
        Vec3f fw = rs.cross(rt);
        if (abs(fw.dot(r)) - tdet * b.heightDiff > fw.abs().dot(boxHalfSize))
            return true;
    }
    else
    {
        rs += m_ns * (b.heightAvg * t.z);
        rt += m_nt * (b.heightAvg * t.z);

//      bounding_volume(s,t,w) =
//          r + n * b.heightDiff * w + rs * s + rt * t +
//          ns *(b.heightDiff * w + b.heightGrad.x * t.z * s) * s +
//          nt *(b.heightDiff * w + b.heightGrad.y * t.z * t) * t +
//          (ns * b.heightGrad.y + nt * b.heightGrad.x) * t.z * s * t

        Vec3f rns = m_ns * (b.heightDiff + abs(b.heightGrad.x) * t.z) * t.z;
        Vec3f rnt = m_nt * (b.heightDiff + abs(b.heightGrad.y) * t.z) * t.z;
        Vec3f rst = (m_ns * b.heightGrad.y + m_nt * b.heightGrad.x) * t.z * t.z;

        // AABB is outside the box => miss.

        quad = rns.abs() + rnt.abs() + rst.abs();
        Vec3f aabb = rs.abs() + rt.abs() + n.abs() * b.heightDiff + quad;
        if ((r.abs() - aabb - boxHalfSize).max() > 0.0f)
            return true;

        // Box is outside the OBB => miss.

        Vec3f fs = n.cross(rt);
        F32 tdet = abs(fs.dot(rs));
        if (abs(fs.dot(r)) - tdet - abs(fs.dot(rns)) - abs(fs.dot(rnt)) - abs(fs.dot(rst)) > fs.abs().dot(boxHalfSize))
            return true;
        Vec3f ft = n.cross(rs);
        if (abs(ft.dot(r)) - tdet - abs(ft.dot(rns)) - abs(ft.dot(rnt)) - abs(ft.dot(rst)) > ft.abs().dot(boxHalfSize))
            return true;
        Vec3f fw = rs.cross(rt);
        if (abs(fw.dot(r)) - tdet * b.heightDiff - abs(fw.dot(rns)) - abs(fw.dot(rnt)) - abs(fw.dot(rst)) > fw.abs().dot(boxHalfSize))
            return true;
    }

    // OBB is small enough => hit.

    Vec3f rw = n.abs() * b.heightDiff;
    F32 ml = max((rs + rt + rw).lenSqr(), (rs - rt + rw).lenSqr(), (rs + rt - rw).lenSqr(), (rs - rt - rw).lenSqr());
    if (sqrt(ml) + quad.length() <= boxHalfSize.length() * (MAX_APPROXIMATION_ERROR) || rect.z <= 0)
    {
        out.rects.add(rect);
        out.area += rs.cross(rt).length() * 8.0f;
        return true;
    }

    // Otherwise => subdivide.

    return false;
}

//------------------------------------------------------------------------

void DisplacedTriangle::intersectBoxTexel(Temp& out, const Vec3f& boxMid, const Vec3f& boxHalfSize, int x, int y) const
{
    Vec2f bary[9];
    Vec2f t((F32)x, (F32)y);
    int vertOfs = out.verts.getSize();

    Quadrangle q;
    getTexelQuadrangle(q, x, y);
    q.p0 -= boxMid;

    if (isectsDeltaTriangleBox(q.p0, q.d1, q.d3, boxHalfSize))
        addVerts(out, t, Vec2f(1.0f, 0.0f), Vec2f(1.0f, 1.0f), clipDeltaTriangleToBox(bary, q.p0, q.d1, q.d3, boxHalfSize), bary);

    if (isectsDeltaTriangleBox(q.p0, q.d3, q.d2, boxHalfSize))
        addVerts(out, t, Vec2f(1.0f, 1.0f), Vec2f(0.0f, 1.0f), clipDeltaTriangleToBox(bary, q.p0, q.d3, q.d2, boxHalfSize), bary);

    if (out.verts.getSize() != vertOfs)
        out.rects.add(Vec3i(x, y, -1));
}

//------------------------------------------------------------------------

void DisplacedTriangle::addVerts(Temp& out, const Vec2f& t, const Vec2f& tu, const Vec2f& tv, int numBary, const Vec2f* bary) const
{
    FW_ASSERT(bary);
    if (numBary < 3)
        return;

    // Transform barys.

    Vec2f v0[12];
    int num = numBary;
    for (int i = 0; i < num; i++)
        v0[i] = t + tu * bary[i].x + tv * bary[i].y;

    // Clip against each edge of the triangle.

    Vec2f v1[12];
    for (int edgeIdx = 0; edgeIdx < 3; edgeIdx++)
    {
        // Select edge and input/output arrays.

        Vec3f e;
        const Vec2f* vin;
        Vec2f* vout;

        switch (edgeIdx)
        {
        case 0:  e = Vec3f(m_bs.x, m_bt.x, m_b.x); vin = v0; vout = v1; break;
        case 1:  e = Vec3f(m_bs.y, m_bt.y, m_b.y); vin = v1; vout = v0; break;
        case 2:  e = Vec3f(-m_bs.x - m_bs.y, -m_bt.x - m_bt.y, 1.0f - m_b.x - m_b.y); vin = v0; vout = v1; break;
        default: FW_ASSERT(false); return;
        }

        // Clip each pair of consecutive vertices.

        int numIn = num;
        const Vec2f* vb = &vin[numIn - 1];
        F32 db = e.x * vb->x + e.y * vb->y + e.z;

        num = 0;
        for (int i = 0; i < numIn; i++)
        {
            const Vec2f* va = vb;
            F32 da = db;
            vb = &vin[i];
            db = e.x * vb->x + e.y * vb->y + e.z;

            if (da >= 0.0f)
                vout[num++] = *va;

            if ((da >= 0.0f) != (db >= 0.0f))
                vout[num++] = lerp(*va, *vb, clamp(da / (da - db), 0.0f, 1.0f));
        }

        // No vertices => done.

        if (!num)
            return;
    }

    // Output vertices and accumulate area.

    Vec2f hi(decFloat(t.x + 1.0f), decFloat(t.y + 1.0f));
    Vec3f pa = 0.0f;
    for (int i = 0; i < num; i++)
    {
        out.verts.add(Vec2f(clamp(v1[i].x, t.x, hi.x), clamp(v1[i].y, t.y, hi.y)));
        Vec3f pb = m_ps * (v1[i].x - v1[0].x) + m_pt * (v1[i].y - v1[0].y);
        out.area += pa.cross(pb).length();
        pa = pb;
    }
}

//------------------------------------------------------------------------

F32 DisplacedTriangle::decFloat(F32 v)
{
    U32 b = floatToBits(v);
    if ((b << 1) == 0)
        b = 0x80000001;
    else if ((b >> 31) == 0)
        b--;
    else if (b < 0xFF7FFFFF)
        b++;
    return bitsToFloat(b);
}

//------------------------------------------------------------------------

void DisplacedTriangle::getNormalizedBounds(DisplacementMap::Bounds& res, const Vec3i& rect) const
{
    FW_ASSERT(m_map);

    // Constant normal => no need to normalize.

    if (m_constantNormal)
    {
        res = m_map->getBounds(rect.x, rect.y, rect.z);
        return;
    }

    // Linear function of normal.

    Vec3f t  = rectToT(rect);
    Vec3f n  = m_n + m_ns * t.x + m_nt * t.y;
    Vec3f ns = m_ns * t.z;
    Vec3f nt = m_nt * t.z;

    // Range of normalization coefficient.

    Vec3f dots(n.dot(ns), n.dot(nt), ns.dot(nt));
    Vec4f corners(+dots.x+dots.y+dots.z, -dots.x+dots.y-dots.z, +dots.x-dots.y-dots.z, -dots.x-dots.y+dots.z);
    F32 clo = 1.0f / sqrt(n.lenSqr() + corners.max() * 2.0f + ns.lenSqr() + nt.lenSqr());
    F32 chi = 1.0f / max(sqrt(n.lenSqr() + corners.min() * 2.0f), m_minNormalLen);
    F32 ca = (clo + chi) * 0.5f;

    // Linear function of displacement factor.

    const DisplacementMap::Bounds& b = m_map->getBounds(rect.x, rect.y, rect.z);
    res.heightGrad = b.heightGrad * ca;
    res.heightAvg = b.heightAvg * ca;
    res.heightDiff = b.heightDiff * chi + (abs(b.heightAvg) + b.heightGrad.abs().dot(Vec2f(t.z))) * (chi - clo) * 0.5f;
}

//------------------------------------------------------------------------

void DisplacedTriangle::getTexelQuadrangle(Quadrangle& res, int x, int y) const
{
    if (m_constantNormal)
    {
        F32 h0 = m_map->getTexel(x, y).height;
        res.p0 = m_n * h0 + m_p + m_ps * (F32)x + m_pt * (F32)y;
        res.d1 = m_n * (m_map->getTexel(x + 1, y).height - h0) + m_ps;
        res.d2 = m_n * (m_map->getTexel(x, y + 1).height - h0) + m_pt;
        res.d3 = m_n * (m_map->getTexel(x + 1, y + 1).height - h0) + m_ps + m_pt;
    }
    else
    {
        Vec2f t((F32)x, (F32)y);
        Vec3f n = m_n + m_ns * t.x + m_nt * t.y;
        Vec3f p = n.normalized(m_map->getTexel(x, y).height);

        res.p0 = p + m_p + m_ps * t.x + m_pt * t.y;
        res.d1 = (n + m_ns).normalized(m_map->getTexel(x + 1, y).height) + m_ps - p;
        res.d2 = (n + m_nt).normalized(m_map->getTexel(x, y + 1).height) + m_pt - p;
        res.d3 = (n + m_ns + m_nt).normalized(m_map->getTexel(x + 1, y + 1).height) + m_ps + m_pt - p;
    }
}

//------------------------------------------------------------------------

void DisplacedTriangle::updateNormal(Normal& res, const Vec3f& t, const Vec2f& g, const Vec2f& ga, const Vec2f& gd, F32 weight) const
{
    // Triangle geometrical normal.

    Vec3f geom = m_ps.cross(m_pt);
    F32 geomLen = geom.length();

    // Triangle shading normal.

    Vec3f n = m_n + m_ns * t.x + m_nt * t.y;
    Vec3f nd = (m_ns.abs() + m_nt.abs()) * t.z;

    // Displaced shading normal.

    F32 det = geom.dot(n);
    F32 flip = ((det >= 0.0f) ? 1.0f : -1.0f);
    Vec3f npt = n.cross(m_pt);
    Vec3f nps = n.cross(m_ps);
    Vec3f ra = n * geomLen;
    Vec3f rd = nd * geomLen;
    rd += npt.abs() * gd.x + nd.cross(m_pt).abs() * (gd.x + abs(ga.x));
    rd += nps.abs() * gd.y + nd.cross(m_ps).abs() * (gd.y + abs(ga.y));
    Vec3f r = ra + (npt * g.x - nps * g.y) * flip;

    Vec3f tt = npt * ga.x - nps * ga.y;
    if (abs(det) >= abs(geom.dot(nd)))
        ra += tt * flip;
    else
        rd += tt.abs();

    // Update result.

    F32 coef = 1.0f / ra.length();
    res.avg += r.normalized(weight);
    res.lo = vecMin(res.lo, (ra - rd) * coef);
    res.hi = vecMin(res.hi, (ra + rd) * coef);
}

//------------------------------------------------------------------------
