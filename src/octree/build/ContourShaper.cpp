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

#include "ContourShaper.hpp"
#include "../Util.hpp"

using namespace FW;

//------------------------------------------------------------------------

ContourShaper::ContourShaper(void)
{
}

//------------------------------------------------------------------------

ContourShaper::~ContourShaper(void)
{
}

//------------------------------------------------------------------------

HullShaper::HullShaper(void)
:   m_mesh              (NULL),
    m_voxelSize         (0.0f),
    m_voxelSizeRcp      (0.0f),

    m_numVoxelTris      (0),
    m_voxelTris         (NULL),
    m_easyCase          (false),
    m_easyRefine        (false),
    m_contour           (0),
    m_numAuxContours    (0)
{
}

//------------------------------------------------------------------------

HullShaper::~HullShaper(void)
{
}

//------------------------------------------------------------------------

void HullShaper::init(const BuilderMeshAccessor* mesh, F32 voxelSize, F32 maxDeviation)
{
    m_mesh = mesh;
    m_voxelSize = voxelSize;
    m_voxelSizeRcp = 1.0f / voxelSize;
    m_devSqr = sqr(maxDeviation);
}

//------------------------------------------------------------------------

void HullShaper::setVoxel(
    const Vec3f&    voxelPos,
    int             numVoxelTris,
    const S32*      voxelTris,
    const S8*       numBary,
    const Vec2f*    bary,
    const S32*      dispIsect,
    int             numAuxContours,
    S32*            auxContours)
{
    F32 baryEps   = 1.0e-6f;
    FW_ASSERT(m_mesh);

    m_numVoxelTris  = numVoxelTris;
    m_voxelTris     = voxelTris;
    m_dispIsect     = dispIsect;
    m_mid           = voxelPos + m_voxelSize * 0.5f;

    // Just the interior of a single triangle => easy case.

    m_easyCase = false;
    if (numVoxelTris == 1 && numBary[0])
    {
        const BuilderMesh::Triangle& tri = m_mesh->getTri(voxelTris[0]);
        bool edge = false;
        for (int i = numBary[0] - 1; i >= 0 && !edge; i--)
        {
            const Vec2f& b = bary[i];
            edge = (min(abs(b.x), abs(b.y), abs(b.x + b.y - 1.0f)) <= baryEps);
        }

        if (!edge && !tri.dispTri)
        {
            // Encode normal.

            m_contour = encodeContourNormal(tri.geomNormal);
            Vec3f normal = decodeContourNormal(m_contour);

            // Encode bounds.

            F32 d = normal.dot(tri.p - m_mid);
            F32 du = normal.dot(tri.pu);
            F32 dv = normal.dot(tri.pv);
            Vec2f b = Vec2f(48.0f, -48.0f) * m_voxelSize;
            for (int i = numBary[0] - 1; i >= 0; i--)
            {
                F32 v = d + du * bary[i].x + dv * bary[i].y;
                b.x = fastMin(b.x, v);
                b.y = fastMax(b.y, v);
            }
            b *= m_voxelSizeRcp;
            m_contour = encodeContourBounds(m_contour, b.x, b.y);
            m_numAuxContours = 0;

            // Need to refine?

            Vec2f posThick = decodeContourPosThick(m_contour);
            F32 error = fastMax(posThick.x - b.x, b.y - posThick.x) + posThick.y * 0.5f;
            m_easyCase = true;
            m_easyRefine = (sqr(error * m_voxelSize) > m_devSqr * normal.lenSqr());
            return;
        }
    }

    // Pick reference planes.

    m_planes.clear();
    const Vec2f* baryPtr = bary;
    const S32* dispIsectPtr = dispIsect;
    int step = max(numVoxelTris >> 4, 1);

    for (int i = 0; i < numVoxelTris; i++)
    {
        const BuilderMesh::Triangle& tri = m_mesh->getTri(voxelTris[i]);

        // Add planes for every nth triangle.

        if (i % step == 0)
        {
            // Displacement => ask DisplacedTriangle.

            if (tri.dispTri)
            {
                m_dispNormals.clear();
                tri.dispTri->getPlaneNormals(m_dispNormals, dispIsectPtr);
                for (int j = 0; j < m_dispNormals.getSize(); j++)
                    addPlane(1.0f, m_dispNormals[j]);
            }
            else
            {
                // Determine boundary mask.

                U8 mask = tri.boundaryMask;
                if (numBary[i])
                {
                    mask = 0;
                    for (int j = numBary[i] - 1; j >= 0; j--)
                    {
                        const Vec2f& b = baryPtr[j];
                        if (abs(b.y) <= baryEps)              mask |= 1;
                        if (abs(b.x + b.y - 1.0f) <= baryEps) mask |= 2;
                        if (abs(b.x) <= baryEps)              mask |= 4;
                    }
                    mask &= tri.boundaryMask;
                }

                // Add planes.

                addPlane(1.0f, tri.geomNormal);
                if ((mask & 1) != 0) addPlane(0.5f, tri.geomNormal.cross(tri.pu));
                if ((mask & 2) != 0) addPlane(0.5f, tri.geomNormal.cross(tri.pv - tri.pu));
                if ((mask & 4) != 0) addPlane(0.5f, tri.geomNormal.cross(tri.pv));
            }
        }

        // Advance to the next triangle.

        if (tri.dispTri)
            dispIsectPtr = DisplacedTriangle::getNextIsect(dispIsectPtr);
        else
            baryPtr += numBary[i];
    }

    // Calculate geometry bounds for the planes.

    baryPtr = bary;
    dispIsectPtr = dispIsect;

    for (int triIdx = 0; triIdx < numVoxelTris; triIdx++)
    {
        const BuilderMesh::Triangle& tri = m_mesh->getTri(voxelTris[triIdx]);

        // Displacement => ask DisplacedTriangle.

        if (tri.dispTri)
        {
            for (int i = m_planes.getSize() - 1; i >= 0; i--)
            {
                Plane& p = m_planes[i];
                tri.dispTri->expandPlaneBounds(p.bounds, dispIsectPtr, p.normal, m_mid);
            }
            dispIsectPtr = DisplacedTriangle::getNextIsect(dispIsectPtr);
            continue;
        }

        // No displacement => check barys.

        Vec3f pp = tri.p - m_mid;
        int num = numBary[triIdx];

        for (int i = m_planes.getSize() - 1; i >= 0; i--)
        {
            Plane& p = m_planes[i];
            F32 d = p.normal.dot(pp);
            F32 du = p.normal.dot(tri.pu);
            F32 dv = p.normal.dot(tri.pv);

            if (!num)
            {
                p.bounds.x = fastMin(p.bounds.x, d + fastMin(0.0f, fastMin(du, dv)));
                p.bounds.y = fastMax(p.bounds.y, d + fastMax(0.0f, fastMax(du, dv)));
            }
            else
            {
                for (int j = num - 1; j >= 0; j--)
                {
                    F32 v = d + du * baryPtr[j].x + dv * baryPtr[j].y;
                    p.bounds.x = fastMin(p.bounds.x, v);
                    p.bounds.y = fastMax(p.bounds.y, v);
                }
            }
        }
        baryPtr += num;
    }

    // Build polyhedron of parent contours.

    m_polyhedron.setCube(-0.5f, 0.5f);
    for (int i = 0; i < numAuxContours; i++)
        isectPolyWithContour(m_polyhedron, auxContours[i], i);

    // Find the best plane and the corresponding contour.

    F32 bestScore = -FW_F32_MAX;
    for (int i = 0; i < m_planes.getSize(); i++)
    {
        const Plane& p = m_planes[i];
        Vec2f geomBounds = p.bounds * m_voxelSizeRcp;
        geomBounds.x = fastMax(geomBounds.x, -48.0f);
        geomBounds.y = fastMin(geomBounds.y, 48.0f);

        // Calculate polyhedron bounds.

        Vec2f polyBounds(48.0f, -48.0f);
        for (int j = 0; j < m_polyhedron.getNumVertices(); j++)
        {
            F32 v = p.normal.dot(m_polyhedron.getVertex(j));
            polyBounds.x = fastMin(polyBounds.x, v);
            polyBounds.y = fastMax(polyBounds.y, v);
        }

        // Evaluate score.

        F32 score = fastMax(polyBounds.y - geomBounds.y, geomBounds.x - polyBounds.x) / p.length * p.weight;
        if (score < bestScore)
            continue;

        // Encode contour.

        bestScore = score;
        m_contour = encodeContourBounds(p.encoded, geomBounds.x, geomBounds.y);
    }

    // Intersect polyhedron.

    isectPolyWithContour(m_polyhedron, m_contour, numAuxContours);

    // Output auxiliary contours.

    U32 auxMask = 0;
    for (int i = 0; i < m_polyhedron.getNumFaces(); i++)
        if (m_polyhedron.getFacePlaneID(i) != -1)
            auxMask |= 1 << m_polyhedron.getFacePlaneID(i);

    auxContours[numAuxContours] = m_contour;
    m_numAuxContours = 0;
    for (int i = 0; i <= numAuxContours; i++)
        if (auxMask & (1 << i))
            auxContours[m_numAuxContours++] = auxContours[i];
}

//------------------------------------------------------------------------

bool HullShaper::needToRefine(void)
{
    FW_ASSERT(m_mesh);
    if (m_easyCase)
        return m_easyRefine;

    m_probes.resize(m_polyhedron.getNumVertices());
    for (int i = 0; i < m_probes.getSize(); i++)
        m_probes[i] = m_polyhedron.getVertex(i) * m_voxelSize + m_mid;

    const S32* dispIsectPtr = m_dispIsect;
    for (int i = 0; i < m_numVoxelTris; i++)
    {
        const BuilderMesh::Triangle& tri = m_mesh->getTri(m_voxelTris[i]);
        if (tri.dispTri)
        {
            tri.dispTri->removeNearbyProbes(m_probes, dispIsectPtr, m_devSqr);
            dispIsectPtr = DisplacedTriangle::getNextIsect(dispIsectPtr);
        }
        else
        {
            for (int j = m_probes.getSize() - 1; j >= 0; j--)
                if (quadrancePointToTri(m_probes[j] - tri.p, tri.pu, tri.pv) <= m_devSqr)
                    m_probes.removeSwap(j);
        }
        if (!m_probes.getSize())
            return false;
    }
    return true;
}

//------------------------------------------------------------------------

void HullShaper::addPlane(F32 weight, const Vec3f& normal)
{
    Plane& p    = m_planes.add();
    p.weight    = weight;
    p.encoded   = encodeContourNormal(normal);
    p.normal    = decodeContourNormal(p.encoded);
    p.length    = p.normal.length();
    p.bounds    = Vec2f(48.0f, -48.0f) * m_voxelSize;

    // Remove duplicates.

    F32 normalThr = 0.98f;
    for (int i = m_planes.getSize() - 2; i >= 0; i--)
    {
        const Plane& ref = m_planes[i];
        if (abs(p.normal.dot(ref.normal)) >= normalThr * p.length * ref.length)
        {
            if (p.weight <= ref.weight)
                m_planes.removeLast();
            else
                m_planes.removeSwap(i);
            break;
        }
    }
}

//------------------------------------------------------------------------
