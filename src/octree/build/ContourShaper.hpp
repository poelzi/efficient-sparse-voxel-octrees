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
#include "BuilderMesh.hpp"
#include "3d/ConvexPolyhedron.hpp"

namespace FW
{
//------------------------------------------------------------------------

class ContourShaper
{
public:
                        ContourShaper       (void);
    virtual             ~ContourShaper      (void);

    virtual void        init                (const BuilderMeshAccessor* mesh, F32 voxelSize, F32 maxDeviation) = 0;
    virtual bool        needsNeighbors      (void) const = 0;

    virtual void        setVoxel            (const Vec3f&   voxelPos,
                                             int            numVoxelTris,
                                             const S32*     voxelTris,
                                             const S8*      numBary,
                                             const Vec2f*   bary,
                                             const S32*     dispIsect,
                                             int            numAuxContours,
                                             S32*           auxContours) = 0; // room for one more aux contour

    virtual void        addNeighbor         (int numVoxelTris, const S32* voxelTris) = 0;

    virtual S32         getContour          (void) = 0;
    virtual int         getNumAuxContours   (void) = 0;
    virtual bool        needToRefine        (void) = 0;

private:
                        ContourShaper       (ContourShaper&); // forbidden
    ContourShaper&      operator=           (ContourShaper&); // forbidden
};

//------------------------------------------------------------------------

class HullShaper : public ContourShaper
{
private:
    struct Plane
    {
        F32             weight;
        S32             encoded;
        Vec3f           normal;
        F32             length;
        Vec2f           bounds;
    };

public:
                        HullShaper          (void);
    virtual             ~HullShaper         (void);

    virtual void        init                (const BuilderMeshAccessor* mesh, F32 voxelSize, F32 maxDeviation);
    virtual bool        needsNeighbors      (void) const    { return false; }

    virtual void        setVoxel            (const Vec3f&   voxelPos,
                                             int            numVoxelTris,
                                             const S32*     voxelTris,
                                             const S8*      numBary,
                                             const Vec2f*   bary,
                                             const S32*     dispIsect,
                                             int            numAuxContours,
                                             S32*           auxContours);

    virtual void        addNeighbor         (int numVoxelTris, const S32* voxelTris) { FW_UNREF(numVoxelTris); FW_UNREF(voxelTris); }

    virtual S32         getContour          (void)          { return m_contour; }
    virtual int         getNumAuxContours   (void)          { return m_numAuxContours; }
    virtual bool        needToRefine        (void);

private:
    void                addPlane            (F32 weight, const Vec3f& normal);

private:
                        HullShaper          (HullShaper&); // forbidden
    HullShaper&         operator=           (HullShaper&); // forbidden

private:
    const BuilderMeshAccessor* m_mesh;
    F32                 m_voxelSize;
    F32                 m_voxelSizeRcp;
    F32                 m_devSqr;

    S32                 m_numVoxelTris;
    const S32*          m_voxelTris;
    const S32*          m_dispIsect;

    Vec3f               m_mid;
    bool                m_easyCase;
    bool                m_easyRefine;
    S32                 m_contour;
    S32                 m_numAuxContours;

    Array<Plane>        m_planes;
    Array<Vec3f>        m_dispNormals;
    ConvexPolyhedron    m_polyhedron;
    Array<Vec3f>        m_probes;
};

//------------------------------------------------------------------------
}
