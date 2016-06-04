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
#include "BuilderBase.hpp"
#include "AttribFilter.hpp"
#include "ContourShaper.hpp"
#include "BuilderMesh.hpp"
#include "base/Hash.hpp"

namespace FW
{
//------------------------------------------------------------------------

class MeshBuilder : public BuilderBase
{
private:
    enum
    {
        GeomExpansion           = 1,                    // sliceBox +- (GeomExpansion << nodeScale)
        GridSizeLog2            = 4,
        GridSize                = 1 << GridSizeLog2
    };

    enum VoxelFlags
    {
        Voxel_RefineGeometry    = 1 << 0,               // Needs more geometry resolution.
        Voxel_RefineAttribs     = 1 << 1,               // Needs more attribute resolution.
        Voxel_InheritAttribs    = 1 << 2,               // Inherits attributes from the parent.
        Voxel_NonExistent       = Voxel_InheritAttribs  // Forbidden value for placeholder voxels that do not really exist.
    };

    enum SpecialParent                                   // Potential return value of processNextParentNode().
    {
        SpecialParent_OutsideSlice  = -1,
        SpecialParent_NotSplit      = -2
    };

    struct VoxelHeader
    {
        Vec3i               pos;
        U16                 flags;
        U16                 parentFlags;
    };

    struct VoxelData
    {
        S32                 numTris;
        S32                 numAuxContours;
        S32                 firstTri;
        S32                 firstBary;
        S32                 firstDispIsect;
        S32                 firstAuxContour;

        Vec3f               colorLo;
        Vec3f               colorHi;
        Vec3f               normalLo;
        Vec3f               normalHi;

        U32                 inheritMask;                                // 3x3x3 bitmask
        U32                 parentInheritMask;                          // 2x2x2 bitmask
        S32                 attribData[8][AttribFilter::DataItem_Max];  // 2x2x2 values
    };

    struct Corner
    {
        bool                exists;
        S32                 data[AttribFilter::DataItem_Max];
        Vec4f               color;
        Vec3f               normal;
    };

    struct Grid
    {
        S32                 base;
        S16                 ofs[GridSize][GridSize][GridSize];
    };

    //------------------------------------------------------------------------

    class ThreadState : public BuilderBase::ThreadState
    {
    public:
                            ThreadState             (MeshBuilder* builder, int idx);
        virtual             ~ThreadState            (void);

    protected:
        virtual void        beginParentSlice        (const Vec3i& cubePos, int cubeScale, int nodeScale, int objectID, int numNodes);
        virtual void        endParentSlice          (void);
        virtual Vec3i       readParentNode          (const Vec3i& cubePos, int cubeScale, int nodeScale);

        virtual void        endChildSlice           (ChildSlice& cs);
        virtual bool        buildChildNode          (ChildSlice& cs, const Vec3i& nodePos);

    private:
        void                createVoxel             (int                voxelIdx,
                                                     int                childIdx,
                                                     const Vec3i&       parentPos,
                                                     U32                parentFlags,
                                                     U32                parentInheritMask,
                                                     const S32          parentAttribData[27][AttribFilter::DataItem_Max]);

        void                sampleAttribs           (TextureSampler::Sample&        color,
                                                     DisplacedTriangle::Normal&     normal,
                                                     const BuilderMesh::Triangle&   tri,
                                                     int                            numBary,
                                                     const Vec2f*                   bary,
                                                     const S32*                     dispIsect);

        Vec3i               processNextParentNode   (void); // May return SpecialParent. Writes m_childNodeExists.

        bool                attachPaletteAttribs    (int voxelIdx, int childIdx, bool isInSlice); // true to refine
        bool                checkPaletteAttribs     (const VoxelData& vd, const Vec4f& color, const Vec3f& normal); // true to refine

        void                attachCornerAttribs     (bool isInSlice);
        bool                checkCornerAttribs      (const VoxelData& vd, const Corner corners[27], int cornerIdx); // true to refine

        void                beginDXTParent          (int voxelIdx, bool isInSlice);
        void                collectDXTVoxel         (int voxelIdx, int childIdx);
        void                flushDXTBlock           (void);

        bool                attachContour           (const VoxelHeader& vh, VoxelData& vd, int childIdx); // true to refine

        void                gridClear               (void);
        Grid*               gridFindOrCreate        (Vec3i gridPos, S32 createForVoxelIdx);
        S32                 gridGet                 (Vec3i scaledPos);
        void                gridSet                 (Vec3i scaledPos, S32 voxelIdx);

    private:
                            ThreadState             (ThreadState&); // forbidden
        ThreadState&        operator=               (ThreadState&); // forbidden

    private:
        MeshBuilder*        m_builder;
        int                 m_threadIdx;            // zero-based

        // Parent slice data.

        Vec3i               m_cubePos;
        S32                 m_cubeScale;
        S32                 m_nodeScale;
        F32                 m_voxelSize;
        Params              m_params;

        BuilderMeshAccessor* m_mesh;

        // Filter and shaper.

        AttribFilter*       m_filter;
        ContourShaper*      m_shaper;

        // Voxels.

        Array<VoxelHeader>  m_voxelHeaders;     // [voxel]
        Array<VoxelData>    m_voxelDatas;       // [voxel]
        Array<S32>          m_voxelTris;        // [vd.firstTri]
        Array<S8>           m_voxelNumBarys;    // [vd.firstTri] zero for full triangles
        Array<Vec2f>        m_voxelBarys;       // [vd.firstBary]
        Array<S32>          m_voxelDispIsect;   // [vd.firstDispIsect]
        Array<S32>          m_voxelAuxContours; // [vd.firstAuxContour]
        Hash<Vec3i, Grid*>  m_gridHash;
        Grid*               m_currGrid;
        Vec3i               m_currGridPos;

        // Voxel iteration.

        S32                 m_currVoxel;
        S32                 m_neighbors[64];
        bool                m_childNodeExists[8];
        const bool*         m_childNodeExistsPtr;

        // DXT data.

        S32                 m_dxtNumParents;
        Vec3i               m_dxtCurrPos;       // Position of the current parent.
        bool                m_dxtIsInSlice;
        S32                 m_dxtVoxels[16];    // Voxel indices, -1 if none.

        // Temporary data.

        Array<S32>          m_parentTris;
        Array<S32>          m_parentDispIsect;
        Array<S32>          m_parentAuxContours;
        DisplacedTriangle::Temp m_dispTemp;
    };

    //------------------------------------------------------------------------

public:
                            MeshBuilder             (OctreeFile* file);
    virtual                 ~MeshBuilder            (void);

    virtual String          getClassName            (void) const                { return "MeshBuilder"; }
    virtual String          getIDString             (void) const                { return "Mesh    "; }
    virtual bool            supportsConcurrency     (void) const                { return true; }

protected:
    virtual bool            createRootSlice         (ChildSlice&                    cs,
                                                     Array<AttachIO::AttachType>&   attach,
                                                     Mat4f&                         octreeToObject,
                                                     int                            objectID,
                                                     const Params&                  params);

    virtual BuilderBase::ThreadState* createThreadState(int idx);
    virtual void            prepareTask             (Task& task);

private:
    static inline bool      componentBelow          (const Vec3f& a, const Vec3f& b) { return (a.x < b.x || a.y < b.y || a.z < b.z); }

    static void             writeParams             (ChildSlice& cs, const Params& params);

    const BuilderMesh*      getMesh                 (int objectID);
    const BuilderMesh*      getOrCreateMesh         (int objectID);

private:
                            MeshBuilder             (MeshBuilder&); // forbidden
    MeshBuilder&            operator=               (MeshBuilder&); // forbidden

private:
    S32                     m_geomExpansionBits;        // ceil(log2(GeomExpansion * 2 + 1))
    S8                      m_lerpCorners[27][5];       // [valueIdx][iter]
    U32                     m_childInheritMasks[8][8];  // [childIdx][cornerIdx]
    U32                     m_neighborInheritMasks[27]; // [neighborIdx]

    Spinlock                m_lock;
    Array<BuilderMesh*>     m_meshes;
};

//------------------------------------------------------------------------
/*

MeshBuilder build data v1
-------------------------

- see BuildDataAttach in BuilderBase.h
- subclassIDString = "Mesh    "

writeBits(32, version); // must be 1
writeBits(32, enableInterpolation);
writeBits(32, enableContours);
writeBits(32, enableVariableResolution);
writeBits(32, floatToBits(colorDeviation));
writeBits(32, floatToBits(normalDeviation));
writeBits(32, floatToBits(contourDeviation));
writeBits(32, filter);

prevTris.clear();
for (int i = 0; i < voxels.getSize(); i++)
{
    // Flags and position.

    writeBits(3, voxels[i].flags);
    for (int j = 0; j < 3; j++)
        writeBits(max(cubeScale - nodeScale, geomExpansionBits) + 1,
            ((voxels[i].pos[j] - cubePos) >> nodeScale) + GeomExpansion);

    // Inherited attributes.

    if ((voxels[i].flags & Voxel_InheritAttribs) != 0)
    {
        writeBits(27, voxels[i].inheritMask);
        for (int j = 0; j < 8; j++)
        {
            if ((voxels[i].inheritMask & (1 << (base2ToBase3(j) * 2))) == 0)
                continue;

            writeBits(32, voxels[i].attribData[j][0]);
            writeBits(32, voxels[i].attribData[j][1]);
        }
    }

    // Triangle indices.

    if (voxels[i].tris == prevTris)
        writeBits(1, 0);
    else
    {
        writeBits(1, 1);
        writeBits(ceil(log2(numTriangles + 1)), voxels[i].tris.getSize());
        for (int j = 0; j < voxels[i].tris.getSize(); j++)
            writeBits(ceil(log2(numTriangles + 1)), voxels[i].tris[j]);
        prevTris = voxels[i].tris;
    }

    // Displacement intersections.

    for (int j = 0; j < voxels[i].tris.getSize(); j++)
        if (voxels[i].tris[j].dispTri)
            voxels[i].tris[j].dispTri.exportIsect();

    // Auxiliary contours.

    for (int j = 0; j < voxels[i].auxContours.getSize(); j++)
    {
        writeBits(1, 1);
        writeBits(32, voxels[i].auxContours[j]);
    }
    writeBits(1, 0);
}
writeBits(3, Voxel_NonExistent);

*/
//------------------------------------------------------------------------
}
