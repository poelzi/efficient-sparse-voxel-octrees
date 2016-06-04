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

#include "AttribFilter.hpp"
#include "BuilderMesh.hpp"
#include "../io/OctreeFile.hpp"
#include "../Util.hpp"

using namespace FW;

//------------------------------------------------------------------------

#define PYRAMID_EXPAND_REL      0.01f

//------------------------------------------------------------------------

void AttribFilter::Value::encode(S32* dataOut, Vec4f& colorOut, Vec3f& normalOut) const
{
    colorOut = getColor() * rcp(getWeight());
    dataOut[DataItem_Color] = colorOut.toABGR();

    normalOut = getNormal().normalized();
    dataOut[DataItem_Normal] = encodeRawNormal(normalOut);
}

//------------------------------------------------------------------------

void AttribFilter::Value::decode(const S32* data)
{
    setWeight(1.0f);
    setColor(Vec4f::fromABGR(data[DataItem_Color]));
    setNormal(Vec3f(decodeRawNormal(data[DataItem_Normal])).normalized());
}

//------------------------------------------------------------------------

AttribFilter::AttribFilter(void)
{
}

//------------------------------------------------------------------------

AttribFilter::~AttribFilter(void)
{
}

//------------------------------------------------------------------------

BoxFilter::BoxFilter(const Vec2i& extent)
:   m_mesh      (NULL),
    m_extent    (extent),
    m_input     (NULL)
{
    FW_ASSERT(extent.x <= 0 && extent.y >= 0);
}

//------------------------------------------------------------------------

BoxFilter::~BoxFilter(void)
{
}

//------------------------------------------------------------------------

void BoxFilter::init(const BuilderMeshAccessor* mesh, F32 voxelSize, int initialCapacity)
{
    FW_UNREF(voxelSize);
    m_mesh = mesh;
    m_voxels.clear();
    m_voxels.setCapacity(initialCapacity);
    m_input = NULL;
}

//------------------------------------------------------------------------

void BoxFilter::inputBegin(int voxelIdx, const Vec3f& voxelPos)
{
    FW_UNREF(voxelPos);
    if (voxelIdx >= m_voxels.getSize())
        m_voxels.resize(voxelIdx + 1);
    m_input = &m_voxels[voxelIdx];
    m_input->clear();
}

//------------------------------------------------------------------------

void BoxFilter::inputTriangle(
    int                                 triIdx,
    const TextureSampler::Sample&       colorSample,
    const DisplacedTriangle::Normal*    normalSample,
    F32                                 dispArea,
    int                                 numBary,
    const Vec2f*                        bary)
{
    FW_ASSERT(m_mesh && m_input && bary);
    const BuilderMesh::Triangle& tri = m_mesh->getTri(triIdx);

    // Displaced triangle.

    if (normalSample)
    {
        F32 weight = max(dispArea, FW_MIN_ATTRIB_WEIGHT);
        m_input->addWeight(weight);
        m_input->addColor(colorSample.avg * weight);
        m_input->addNormal(normalSample->avg * weight);
        return;
    }

    // Unclipped triangle.

    if (!numBary)
    {
        if (!tri.colorTex && !tri.alphaTex)
            *m_input += tri.average;
        else
        {
            F32 weight = max(tri.area * colorSample.avg.w, FW_MIN_ATTRIB_WEIGHT);
            m_input->addWeight(weight);
            m_input->addColor(colorSample.avg * weight);
            m_input->addNormal(tri.avgNormal * weight);
        }
        return;
    }

    // Clipped triangle.

    FW_ASSERT(numBary >= 3);
    Vec2f baryAvg = bary[numBary - 1];
    F32 relArea = baryAvg.cross(bary[0]);
    for (int i = numBary - 2; i >= 0; i--)
    {
        baryAvg += bary[i];
        relArea += bary[i].cross(bary[i + 1]);
    }
    baryAvg *= 1.0f / (F32)numBary;

    F32 weight = max(tri.area * relArea * colorSample.avg.w, FW_MIN_ATTRIB_WEIGHT);
    m_input->addWeight(weight);
    m_input->addColor(colorSample.avg * weight);
    m_input->addNormal((tri.n + tri.nu * baryAvg.x + tri.nv * baryAvg.y) * weight);
}

//------------------------------------------------------------------------

PyramidFilter::PyramidFilter(void)
:   m_mesh          (NULL),
    m_voxelSize     (0.0f),
    m_weightCoef    (0.0f),
    m_input         (NULL)
{
}

//------------------------------------------------------------------------

PyramidFilter::~PyramidFilter(void)
{
}

//------------------------------------------------------------------------

void PyramidFilter::init(const BuilderMeshAccessor* mesh, F32 voxelSize, int initialCapacity)
{
    m_mesh = mesh;
    m_voxels.clear();
    m_voxels.setCapacity(initialCapacity);
    m_voxelSize = voxelSize;
    m_weightCoef = pow(voxelSize, -4.0f);
    m_input = NULL;
}

//------------------------------------------------------------------------

void PyramidFilter::inputBegin(int voxelIdx, const Vec3f& voxelPos)
{
    if (voxelIdx >= m_voxels.getSize())
        m_voxels.resize(voxelIdx + 1);
    m_input = &m_voxels[voxelIdx];
    memset(m_input->v, 0, sizeof(m_input->v));
    m_voxelPos = voxelPos - m_voxelSize * (PYRAMID_EXPAND_REL * 0.5f);
}

//------------------------------------------------------------------------

void PyramidFilter::inputTriangle(
    int                                 triIdx,
    const TextureSampler::Sample&       colorSample,
    const DisplacedTriangle::Normal*    normalSample,
    F32                                 dispArea,
    int                                 numBary,
    const Vec2f*                        bary)
{
    FW_ASSERT(m_mesh && m_input);
    FW_UNREF(dispArea);
    if (normalSample)
        fail("PyramidFilter does not support displacement maps!");

    const BuilderMesh::Triangle& tri = m_mesh->getTri(triIdx);
    F32 weight = tri.area * colorSample.avg.w * m_weightCoef;

    if (!numBary)
    {
        static const Vec2f unclippedBary[] = { Vec2f(0.0f, 0.0f), Vec2f(1.0f, 0.0f), Vec2f(0.0f, 1.0f) };
        numBary = 3;
        bary = unclippedBary;
    }
    FW_ASSERT(numBary >= 3);

    const Vec2f& b = bary[0];
    Vec3f p = tri.p + tri.pu * b.x + tri.pv * b.y - m_voxelPos;
    Vec3f n = tri.n + tri.nu * b.x + tri.nv * b.y;

    Vec2f bv = bary[numBary - 1] - b;
    Vec3f pv = tri.pu * bv.x + tri.pv * bv.y;
    Vec3f nv = tri.nu * bv.x + tri.nv * bv.y;

    for (int i = numBary - 2; i > 0; i--)
    {
        Vec2f bu = bary[i] - b;
        Vec3f pu = tri.pu * bu.x + tri.pv * bu.y;
        Vec3f nu = tri.nu * bu.x + tri.nv * bu.y;
        setTriangle(max(weight * bu.cross(bv), FW_MIN_ATTRIB_WEIGHT), p, pu, pv);

        for (int j = 0; j < 8; j++)
        {
            F32* v                  = m_input->v[j];
            const Vec3f& c          = m_coefs[j];
            v[Component_Weight]     += c.x;
            v[Component_ColorR]     += c.x * colorSample.avg.x;
            v[Component_ColorG]     += c.x * colorSample.avg.y;
            v[Component_ColorB]     += c.x * colorSample.avg.z;
            v[Component_ColorA]     += c.x * colorSample.avg.w;
            v[Component_NormalX]    += c.x * n.x + c.y * nu.x + c.z * nv.x;
            v[Component_NormalY]    += c.x * n.y + c.y * nu.y + c.z * nv.y;
            v[Component_NormalZ]    += c.x * n.z + c.y * nu.z + c.z * nv.z;
        }

        bv = bu;
        pv = pu;
        nv = nu;
    }
}

//------------------------------------------------------------------------

void PyramidFilter::outputAccumulate(int voxelIdx, const Vec3i& posInNeighborhood)
{
    const Voxel& v = m_voxels[voxelIdx];
    const Vec3i& p = posInNeighborhood;
    F32 bias = m_voxelSize * (PYRAMID_EXPAND_REL + 1.0f);

    switch (p.x + p.y * 2 + p.z * 4)
    {
#define CASE(N, X) case N: for (int i = 0; i < Component_Max; i++) m_output[i] += X; break;
        CASE(0, v.v[0][i])
        CASE(1, v.v[1][i] * bias - v.v[0][i])
        CASE(2, v.v[2][i] * bias - v.v[0][i])
        CASE(3, v.v[0][i] - (v.v[1][i] + v.v[2][i] - v.v[3][i] * bias) * bias)
        CASE(4, v.v[4][i] * bias - v.v[0][i])
        CASE(5, v.v[0][i] - (v.v[1][i] + v.v[4][i] - v.v[5][i] * bias) * bias)
        CASE(6, v.v[0][i] - (v.v[2][i] + v.v[4][i] - v.v[6][i] * bias) * bias)
        CASE(7, (v.v[1][i] + v.v[2][i] + v.v[4][i] - (v.v[3][i] + v.v[5][i] + v.v[6][i] - v.v[7][i] * bias) * bias) * bias - v.v[0][i])
#undef CASE
    }
}

//------------------------------------------------------------------------

void PyramidFilter::setTriangle(F32 weight, const Vec3f& p, const Vec3f& pu, const Vec3f& pv)
{
    m_w[0][0] =             weight * 180.0f;
    m_w[1][0] = m_w[0][1] = weight * 60.0f;
    m_w[2][0] = m_w[0][2] = weight * 30.0f;
    m_w[1][1] =             weight * 15.0f;
    m_w[3][0] = m_w[0][3] = weight * 18.0f;
    m_w[2][1] = m_w[1][2] = weight * 6.0f;
    m_w[4][0] = m_w[0][4] = weight * 12.0f;
    m_w[3][1] = m_w[1][3] = weight * 3.0f;
    m_w[2][2] =             weight * 2.0f;

#define UV(U,V) \
    m_zw[U][V] = p.z * m_w[U][V] + pu.z * m_w[U+1][V] + pv.z * m_w[U][V+1];
    UV(0,0) UV(1,0) UV(2,0) UV(3,0) UV(0,1) UV(1,1) UV(2,1) UV(0,2) UV(1,2) UV(0,3)
#undef UV

#define UV(U,V) \
    m_yw[U][V] = p.y * m_w[U][V] + pu.y * m_w[U+1][V] + pv.y * m_w[U][V+1]; \
    m_yzw[U][V] = p.y * m_zw[U][V] + pu.y * m_zw[U+1][V] + pv.y * m_zw[U][V+1];
    UV(0,0) UV(1,0) UV(2,0) UV(0,1) UV(1,1) UV(0,2)
#undef UV

#define UV(C,U,V) \
    m_coefs[0].C = p.x * m_yzw[U][V] + pu.x * m_yzw[U+1][V] + pv.x * m_yzw[U][V+1]; \
    m_coefs[1].C = m_yzw[U][V]; \
    m_coefs[2].C = p.x * m_zw[U][V] + pu.x * m_zw[U+1][V] + pv.x * m_zw[U][V+1]; \
    m_coefs[3].C = m_zw[U][V]; \
    m_coefs[4].C = p.x * m_yw[U][V] + pu.x * m_yw[U+1][V] + pv.x * m_yw[U][V+1]; \
    m_coefs[5].C = m_yw[U][V]; \
    m_coefs[6].C = p.x * m_w[U][V] + pu.x * m_w[U+1][V] + pv.x * m_w[U][V+1]; \
    m_coefs[7].C = m_w[U][V];
    UV(x,0,0) UV(y,1,0) UV(z,0,1)
#undef UV
}

//------------------------------------------------------------------------
